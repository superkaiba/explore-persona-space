"""Shared configuration + helpers for issue #1902 (OLMo-2 post-training stage map).

Single source of truth for every constant the #1902 pipeline shares across
phases (P0 corpus build, P1 pilot, P2 generation, P3 capture, P4 fits, P5
figures): model ids + revision pins, the capture-layer set, corpus schemas,
HF data-repo layout, prompt-render helpers, and the P2 degeneracy filters.

Design rules baked in (plan #1902 v4):

- Model dims (n_layers, hidden) are ALWAYS taken from ``AutoConfig`` at run
  time via :func:`model_dims` — never hardcoded (the parent issue scripts'
  Qwen-shaped layer/width constants must not leak here; plan A6).
- Revision pins are BINDING once P1 writes them to ``pilot_report.json``:
  every later model/tokenizer load passes ``revision=<pinned sha>``
  (:func:`load_revision_pins` / :func:`resolve_revision`).
- Context/prefix activation summaries are stored ONCE per (checkpoint,
  corpus) under ``<ckpt>/ctx/<corpus>/L{l}.pt``; answer summaries per
  (checkpoint, answer-source, corpus) under ``<ckpt>/<src>/<corpus>/L{l}.pt``
  (:func:`ctx_store_relpath` / :func:`answer_store_relpath`).

VM-run thread-cap prefix (shared VM; #847/#891/#1315) — every VM-side launch
of a #1902 script carries::

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue1902_corpus.py --full

Content hygiene: LMSYS is unscreened real user text — no function in this
module (or its consumers) may print/log corpus row text; digests are
filename + index + counts only.
"""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import NamedTuple

# ── checkpoints / model ids ──────────────────────────────────────────────────

# Stage order is B → S → D → R (base → SFT → DPO → RLVR/Instruct).
CKPTS: tuple[str, ...] = ("B", "S", "D", "R")

MODEL_IDS: dict[str, str] = {
    "B": "allenai/OLMo-2-1124-7B",
    "S": "allenai/OLMo-2-1124-7B-SFT",
    "D": "allenai/OLMo-2-1124-7B-DPO",
    "R": "allenai/OLMo-2-1124-7B-Instruct",
}

# Smoke-only model remap (the tiny-real standard, #906): when set, EVERY
# checkpoint id resolves to ONE local directory holding a tiny random-weights
# Olmo2 model over the REAL vocab (real tokenizer). This is an env-gated
# model-ID substitution, never a code-path fork — every phase body runs
# byte-identically. Production never sets it.
_SMOKE_MODEL_DIR = os.environ.get("EPM_ISSUE1902_SMOKE_MODEL_DIR")
if _SMOKE_MODEL_DIR:
    MODEL_IDS = {c: _SMOKE_MODEL_DIR for c in CKPTS}

# Name of the P1 pilot report that carries the BINDING revision pins
# (plan §10: pinned at P1 launch; every P2/P3/robustness load passes
# revision=<pinned sha> explicitly).
PILOT_REPORT_NAME = "pilot_report.json"
REVISION_PINS_KEY = "revision_pins"


def default_revision_pins() -> dict[str, str | None]:
    """Unpinned placeholder pins (pre-P1 only — e.g. the P0 tokenizer load)."""
    return {c: None for c in CKPTS}


def revision_pins_from_report(report: dict) -> dict[str, str]:
    """Extract + validate the binding revision-pin dict from a pilot report."""
    pins = report.get(REVISION_PINS_KEY) or {}
    missing = [c for c in CKPTS if not pins.get(c)]
    if missing:
        raise RuntimeError(
            f"pilot report is missing revision pins for checkpoints {missing}; "
            f"expected key {REVISION_PINS_KEY!r} with all of {CKPTS}"
        )
    return {c: str(pins[c]) for c in CKPTS}


def load_revision_pins(pilot_report_path: Path | str) -> dict[str, str]:
    """Load the binding per-checkpoint revision pins from ``pilot_report.json``."""
    path = Path(pilot_report_path)
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    return revision_pins_from_report(report)


def resolve_revision(ckpt: str, pins: dict[str, str | None] | None) -> str | None:
    """Revision to pass to a HF load for ``ckpt`` (None = unpinned, pre-P1 only).

    A ``local:`` pin (smoke model-dir remap) resolves to None — HF revision
    kwargs are meaningless for a local directory load.
    """
    _check_ckpt(ckpt)
    if pins is None:
        return None
    pin = pins.get(ckpt)
    if pin is not None and str(pin).startswith("local:"):
        return None
    return pin


def pin_revisions_now() -> dict[str, str]:
    """Resolve each checkpoint repo's CURRENT main sha (P1 calls this ONCE at
    launch and persists the result into ``pilot_report.json``).

    Local-directory model ids (the smoke remap) pin to a ``local:<sha16>`` of
    the directory's config.json so the pin stays content-addressed without a
    Hub call."""
    import hashlib

    from huggingface_hub import HfApi

    pins: dict[str, str] = {}
    api = HfApi()
    for c in CKPTS:
        mid = MODEL_IDS[c]
        if Path(mid).is_dir():
            digest = hashlib.sha256((Path(mid) / "config.json").read_bytes()).hexdigest()[:16]
            pins[c] = f"local:{digest}"
        else:
            pins[c] = str(api.model_info(mid).sha)
    return pins


# ── model dims (AutoConfig-derived — NEVER hardcoded) ────────────────────────


class ModelDims(NamedTuple):
    num_layers: int
    hidden_size: int
    max_position_embeddings: int
    vocab_size: int


def model_dims(model_id: str, revision: str | None = None) -> ModelDims:
    """(n_layers, d, max_pos, vocab) from AutoConfig at run time.

    Plan A6: the parent Qwen scripts carry module-level Qwen-shaped
    layer/width constants; #1902 code derives dims from the checkpoint config.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id, revision=revision)
    return ModelDims(
        num_layers=int(cfg.num_hidden_layers),
        hidden_size=int(cfg.hidden_size),
        max_position_embeddings=int(cfg.max_position_embeddings),
        vocab_size=int(cfg.vocab_size),
    )


def capture_layers(num_layers: int) -> tuple[int, ...]:
    """Capture-layer set: every 2nd layer plus the final layer.

    For the OLMo-2 32-layer chain this is {0, 2, 4, ..., 30, 31} — 17 layers
    (plan §4 P3). Derived from ``num_layers`` so a different-depth checkpoint
    can never silently reuse a stale hardcoded set.
    """
    if num_layers < 2:
        raise ValueError(f"num_layers must be >= 2, got {num_layers}")
    return tuple(sorted(set(range(0, num_layers, 2)) | {num_layers - 1}))


# ── HF data-repo layout ──────────────────────────────────────────────────────

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1902_stage_map"
# READ paths (corpus staging) stay on the production prefix; WRITE paths
# (rollouts, store, eval mirror, pilot timing) divert under
# EPM_ISSUE1902_HF_WRITE_PREFIX so a smoke run NEVER overwrites production
# artifacts (the dispatcher exports `<prefix>/_smoke` under --smoke).
HF_WRITE_PREFIX = os.environ.get("EPM_ISSUE1902_HF_WRITE_PREFIX", HF_PREFIX)
CORPUS_HF_PATH = f"{HF_PREFIX}/corpus"
RAW_GEN_HF_PATH = f"{HF_WRITE_PREFIX}/raw_completions/gen"  # + /{single|multi}/{ckpt}.json
STORE_HF_PATH = f"{HF_WRITE_PREFIX}/analysis_tensors/issue1902_store"
EVAL_MIRROR_HF_PATH = f"{HF_WRITE_PREFIX}/eval_results_mirror"
PILOT_TIMING_HF_PATH = f"{HF_WRITE_PREFIX}/pilot_timing/shard"

# ── corpora ──────────────────────────────────────────────────────────────────

CORPUS_SINGLE = "single"
CORPUS_MULTI = "multi"
CORPORA: tuple[str, ...] = (CORPUS_SINGLE, CORPUS_MULTI)

CORPUS_SINGLE_FILENAME = "corpus_single.jsonl"
CORPUS_MULTI_FILENAME = "corpus_multi.jsonl"
CLUSTERS_FILENAME = "clusters.json"

# Source datasets (revisions pinned for deterministic stream order + resume
# fingerprints; the LMSYS pin is the #1092-verified sha).
LMSYS_DATASET = "lmsys/lmsys-chat-1m"
LMSYS_REVISION = "200748d9d3cddcc9d782887541057aca0b18c5da"
GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_REVISION = "740312add88f781978c0658806c59bc2815b9866"
MBPP_DATASET = "google-research-datasets/mbpp"
MBPP_CONFIG = "full"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"

# LMSYS stores FULL language names ('English', not 'en') — gotchas real-corpus
# entry / plan A20.
LANG_FILTER = "English"

# Draw quotas (plan §4 P0).
SINGLE_GENERIC_N = 16_000
SINGLE_MATHCODE_N = 1_200
GSM8K_N = 500
MBPP_N = 300
MULTI_N = 16_000
SCAN_CAP = 800_000  # total streamed-row cap across both corpora (one shared pass)

# Intersection survival targets (plan §4 P2 / gate A').
INTERSECTION_TARGET = 8_192  # = 2d
INTERSECTION_FLOOR = 5_000

# Clustering (plan §4 P0).
K_CLUSTERS = 40
KMEANS_SEED = 42
FOLD_SEED = 42
N_FOLDS = 6
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_BATCH_SIZE = 256

# Formatted-prompt token budget: max_pos 4096 − max_new_tokens 1024 (plan §4
# P0; the #952 load-time length-validation rule — tokenize the FORMATTED
# render, filter under the LONGEST render).
MAX_FORMATTED_TOKENS = 3_072

# Keyword screen for the LMSYS math/code/reasoning enrichment stratum
# (substring match on the lowercased query; enrichment only, pilot-tunable).
MATHCODE_KEYWORDS: tuple[str, ...] = (
    "python",
    "javascript",
    "typescript",
    " java ",
    "c++",
    " sql",
    "regex",
    "algorithm",
    "function that",
    "write a function",
    "write code",
    "write a program",
    "debug",
    "compile",
    "recursion",
    "dataframe",
    "numpy",
    "pandas",
    "equation",
    "solve for",
    "integral",
    "derivative",
    "theorem",
    "proof",
    "matrix",
    "probability",
    "algebra",
    "geometry",
    "calculate",
    "arithmetic",
    "math problem",
)

# Marked tier-2 strata class/group labels (excluded from generic clusters;
# cluster id −1 by convention).
CLASS_GENERIC = "generic"
CLASS_MATHCODE = "mathcode"
CLASS_GSM8K = "gsm8k"
CLASS_MBPP = "mbpp"
UNCLUSTERED = -1

# Corpus row schemas (JSONL; one row per context cell). ``class`` is one of
# the CLASS_* labels; ``group`` is the group-fold axis label — ``cluster_<k>``
# for LMSYS rows, the stratum name for marked strata; ``cluster`` is the
# k-means id (UNCLUSTERED for marked strata / probe runs); ``source_index``
# is the row's position in the pinned source stream (provenance, never text).
SINGLE_ROW_FIELDS: tuple[str, ...] = (
    "id",
    "corpus",
    "class",
    "group",
    "cluster",
    "query",
    "n_tokens_chat",
    "n_tokens_plain",
    "source",
    "source_index",
)
MULTI_ROW_FIELDS: tuple[str, ...] = (
    "id",
    "corpus",
    "class",
    "group",
    "cluster",
    "prefix_turns",
    "query",
    "n_prior_turns",
    "n_tokens_chat",
    "n_tokens_plain",
    "source",
    "source_index",
)

# ── activation-store layout (plan §4 P3) ─────────────────────────────────────

CTX_SOURCE = "ctx"  # reserved answer-source slot name for context summaries


def _check_ckpt(ckpt: str) -> None:
    if ckpt not in CKPTS:
        raise ValueError(f"unknown checkpoint {ckpt!r}; expected one of {CKPTS}")


def _check_corpus(corpus: str) -> None:
    if corpus not in CORPORA:
        raise ValueError(f"unknown corpus {corpus!r}; expected one of {CORPORA}")


def ctx_store_relpath(ckpt: str, corpus: str, layer: int) -> str:
    """Context/prefix summary shard — stored ONCE per (ckpt, corpus).

    Context summaries are identical across the answer-source axis (causal
    attention: answer tokens sit after the context positions), so they are
    deduplicated out of the per-cell layout.
    """
    _check_ckpt(ckpt)
    _check_corpus(corpus)
    return f"{ckpt}/{CTX_SOURCE}/{corpus}/L{int(layer)}.pt"


def answer_store_relpath(ckpt: str, src: str, corpus: str, layer: int) -> str:
    """Answer-summary shard for grid cell (activation ckpt, answer source)."""
    _check_ckpt(ckpt)
    _check_ckpt(src)
    _check_corpus(corpus)
    return f"{ckpt}/{src}/{corpus}/L{int(layer)}.pt"


def cell_row_index_relpath(ckpt: str, src: str, corpus: str) -> str:
    """Per-cell row-index manifest (uploaded with the store — #825 lesson)."""
    _check_ckpt(ckpt)
    _check_corpus(corpus)
    if src != CTX_SOURCE:
        _check_ckpt(src)
    return f"{ckpt}/{src}/{corpus}/row_index.jsonl"


def store_local_path(store_root: Path | str, relpath: str) -> Path:
    return Path(store_root) / relpath


# ── prompt renders (plan §4 P2/P3) ───────────────────────────────────────────

# Stop sequences for the base checkpoint's plain-QA generation render
# (the #825 naturalistic-render precedent).
PLAIN_STOP_SEQUENCES: tuple[str, ...] = ("\nUser:", "User:")


def _validated_prefix_turns(prefix_turns: list[dict] | None) -> list[dict]:
    turns = prefix_turns or []
    for t in turns:
        role = t.get("role")
        if role not in ("user", "assistant"):
            raise ValueError(f"unexpected prefix-turn role {role!r}")
        if not t.get("content"):
            raise ValueError("empty prefix-turn content")
    return turns


def render_plain_prompt(query: str, prefix_turns: list[dict] | None = None) -> str:
    """Plain-text generation/capture prompt: ``User: {q}\\nAssistant:``.

    Multi-turn: prefix turns render as alternating ``User:``/``Assistant:``
    lines before the final user query.
    """
    lines: list[str] = []
    for t in _validated_prefix_turns(prefix_turns):
        label = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{label}: {t['content']}")
    lines.append(f"User: {query}")
    lines.append("Assistant:")
    return "\n".join(lines)


def render_plain_full(query: str, answer: str, prefix_turns: list[dict] | None = None) -> str:
    """Canonical teacher-forcing render: ``User: {q}\\nAssistant: {a}`` (plan §4 P3)."""
    return render_plain_prompt(query, prefix_turns) + f" {answer}"


def chat_messages(query: str, prefix_turns: list[dict] | None = None) -> list[dict]:
    """Message list for ``tokenizer.apply_chat_template`` (S/D/R native render)."""
    msgs = [
        {"role": t["role"], "content": t["content"]} for t in _validated_prefix_turns(prefix_turns)
    ]
    msgs.append({"role": "user", "content": query})
    return msgs


def render_chat_prompt(tokenizer, query: str, prefix_turns: list[dict] | None = None) -> str:
    """Native chat-template generation prompt for the S/D/R checkpoints."""
    return tokenizer.apply_chat_template(
        chat_messages(query, prefix_turns),
        tokenize=False,
        add_generation_prompt=True,
    )


# ── shared-node GPU sizing (fellows H200 hosts share nodes WITHOUT GPU
#    isolation — every device can carry other tenants' memory; #1902 crash 1) ─
# The vLLM gpu_memory_utilization resolver (vllm_util_for_free / the cap,
# margin, and floor constants) is hoisted to the shared module
# ``explore_persona_space.eval.vllm_util`` (#1942); import from there.


def realized_gpu_ids(env, detected: int) -> tuple[str, list[str]]:
    """Realized GPU width + PHYSICAL device-id list — SLURM allocation FIRST.

    On a SLURM job (``SLURM_JOB_ID`` set) the fellows cluster shares nodes
    without GPU isolation: ``nvidia-smi -L`` / torch enumerate the PHYSICAL
    node (8× H200) — and nvidia-smi ignores ``CUDA_VISIBLE_DEVICES`` entirely
    — so a bare detected count over-shards onto other tenants' devices
    (#1902 crash 1: ``[dispatch] ... ngpu=8`` on a 4-GPU allocation).
    Preference order inside a SLURM job:

    1. ``CUDA_VISIBLE_DEVICES`` (slurm-set) — authoritative id list.
    2. ``SLURM_JOB_GPUS`` / ``SLURM_STEP_GPUS`` — the allocation's physical ids.
    3. ``SLURM_GPUS_ON_NODE`` — count only; ids ASSUMED ``0..N-1`` (no id
       source exists in this configuration; the dispatcher logs the source
       token so the assumption is visible in the launch log).

    The id list is CLAMPED to ``SLURM_GPUS_ON_NODE`` (the sbatch template
    asserts it equals the requested width) when present. A SLURM job with
    NONE of the three vars fails loud — never the physical count. Non-SLURM
    lanes (RunPod/GCE exclusive hosts) keep the detected count, ids
    ``0..detected-1``.

    Returns ``(source_token, ids)``; ids are strings (CVD values verbatim).
    """

    def _split(v: str) -> list[str]:
        return [t for t in v.replace(",", " ").split() if t]

    if env.get("SLURM_JOB_ID"):
        ids: list[str] | None = None
        src = ""
        cvd = env.get("CUDA_VISIBLE_DEVICES")
        job_gpus = env.get("SLURM_JOB_GPUS") or env.get("SLURM_STEP_GPUS")
        if cvd:
            ids, src = _split(cvd), "slurm-cvd"
        elif job_gpus:
            ids, src = _split(job_gpus), "slurm-job-gpus"
        elif env.get("SLURM_GPUS_ON_NODE"):
            n = int(env["SLURM_GPUS_ON_NODE"])
            ids, src = [str(i) for i in range(n)], "slurm-count-ids-assumed-0..N-1"
        if not ids:
            raise RuntimeError(
                "SLURM job env carries none of CUDA_VISIBLE_DEVICES / "
                "SLURM_JOB_GPUS / SLURM_STEP_GPUS / SLURM_GPUS_ON_NODE — "
                "refusing to fall back to the physical nvidia-smi count on a "
                "shared node (#1902 crash 1)"
            )
        n_req = env.get("SLURM_GPUS_ON_NODE")
        if n_req and len(ids) > int(n_req):
            ids = ids[: int(n_req)]
            src += "-clamped"
        return src, ids
    n = max(1, int(detected))
    return "detected", [str(i) for i in range(n)]


# ── generation constants (plan §4 P2 — verbatim #779 protocol) ───────────────

GEN_MAX_TOKENS = 1_024
GEN_TEMPERATURE = 1.0
GEN_TOP_P = 0.95
GEN_SEED = 42
RELIABILITY_SEEDS: tuple[int, ...] = (43, 44)
RELIABILITY_SUBSET_N = 1_000

# ── degeneracy filters (plan §4 P2 — symmetric across all four sources) ──────

# Repetition-loop flag: any REPEAT_NGRAM_N-gram (whitespace-token level)
# repeated >= REPEAT_NGRAM_MIN_COUNT times within one response.
# Both constants are `ungrounded — needs smoke-test` in the plan: pilot-tuned.
REPEAT_NGRAM_N = 8
REPEAT_NGRAM_MIN_COUNT = 8


def is_truncated(n_generated_tokens: int, cap: int = GEN_MAX_TOKENS) -> bool:
    """Truncation flag: the generation hit the max_tokens cap (plan §4 P2)."""
    return n_generated_tokens >= cap


def has_repetition_loop(
    text: str,
    n: int = REPEAT_NGRAM_N,
    min_count: int = REPEAT_NGRAM_MIN_COUNT,
) -> bool:
    """Repetition-loop flag: any ``n``-gram repeated >= ``min_count`` times.

    Whitespace-token n-grams (counted over all sliding positions, overlaps
    included). Cheap, deterministic, text-only — never logs the text.
    """
    toks = text.split()
    if len(toks) < n:
        return False
    counts: Counter[tuple[str, ...]] = Counter(
        tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)
    )
    return bool(counts) and max(counts.values()) >= min_count
