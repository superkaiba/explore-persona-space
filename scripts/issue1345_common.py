"""Issue #1345 — shared constants + helpers for the cross-framing operator study.

Is the assistant context->answer map the SAME linear operator across three
framings (chat template / plain User:/Assistant: / assistant-in-narrative
stories)? This module carries the cell/pair registry, the pinned parent
artifact locations (#825 S-track @ 7159e5804d), the prefix-slot render
wrappers (the ONE extraction delta vs #825, plan §4 Phase 2a), and the story
parsing + per-turn render for the R3 regime.

Everything heavy is IMPORTED from the #825 modules (issue825_render_formats,
issue825_extract_turnstore, issue825_fit_cells, issue825_crossmodel_map_transfer,
issue825_map_alignment) — never copied (plan §2 Infrastructure reuse).

Content hygiene: the R1/R2 corpus is LMSYS-derived real user text — helpers
here never print prompt/story text; digests (counts, ids, hashes) only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import issue825_render_formats as rf  # noqa: E402

from explore_persona_space.experiments.issue_825.common import Rendered  # noqa: E402

# ---------------------------------------------------------------------------
# Pinned parent artifacts (plan §10; verified 2026-07-15: all four stems have
# 10 .pt + 10 .json shards, and the track-S corpus JSONL resolves at this rev)
# ---------------------------------------------------------------------------
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
PIN_REV = "7159e5804d"  # the commit where the naturalistic_s upload landed
PARENT_TENSOR_PREFIX = "issue825_userbase_map/analysis_tensors"
PARENT_TRACK_S_JSONL = "issue825_userbase_map/raw_completions/track_s/track_s.jsonl"
PARENT_STEMS = (
    "instruct_chat_s",
    "pretrained_chat_s",
    "instruct_naturalistic_s",
    "pretrained_naturalistic_s",
)

# Upload destinations for THIS issue (issueN_<slug> prefix per Upload Policy;
# plan §10 wrote a bare `analysis_tensors/issue_1345/...` — normalized to the
# canonical issueN-prefixed layout, flagged in the implementation report).
HF_ISSUE_PREFIX = "issue1345_framing"
HF_TENSOR_PREFIX = f"{HF_ISSUE_PREFIX}/analysis_tensors"
HF_STORIES_PREFIX = f"{HF_ISSUE_PREFIX}/raw_completions/stories"

# ---------------------------------------------------------------------------
# Local layout (repo-relative; the dispatcher cds to repo root)
# ---------------------------------------------------------------------------
DATA_DIR = Path("data/issue_1345")
TURNSTORE_DIR = DATA_DIR / "turnstore"
STORIES_DIR = DATA_DIR / "stories"
MATCHED_DIR = DATA_DIR / "matched_n"
PREDS_CACHE_DIR = DATA_DIR / "preds_cache"
PARENT_DL_DIR = DATA_DIR / "hf_dl"
EVAL_DIR = Path("eval_results/issue_1345")
FIG_DIR = Path("figures/issue_1345")

# ---------------------------------------------------------------------------
# Registry: 3 regimes x 2 models x 2 arms (single source for EVERY phase —
# fits enumerate cells, transfer enumerates ordered pairs, operator comparison
# enumerates unordered pairs; smoke thins ROWS, never this registry)
# ---------------------------------------------------------------------------
MODELS = ("instruct", "pretrained")
MODEL_SLUG = {"instruct": "instruct", "pretrained": "base"}  # plan §6.5 file slugs
REGIMES = ("r1", "r2", "r3")
REGIME_FORMAT = {"r1": "chat", "r2": "naturalistic", "r3": "stories"}
ARMS = ("prefix", "context")
# Slot order in the #1345 stores: the extractor sorts slots by token position
# and the prefix slot always precedes the context slot (asserted at render).
ARM_SLOT_INDEX = {"prefix": 0, "context": 1}
# Turn order: R1/R2 single-turn track-S spans sort [u1, a1] -> target = 1;
# R3 rows carry a single "answer" span -> target = 0.
TARGET_TURN_INDEX = {"r1": 1, "r2": 1, "r3": 0}
TRACK = "s"

ORDERED_PAIRS = [(i, j) for i in REGIMES for j in REGIMES if i != j]
UNORDERED_PAIRS = [("r1", "r2"), ("r1", "r3"), ("r2", "r3")]
PAIRED_PAIR = ("r1", "r2")  # the only conv_id-paired pair (reparam leg)


def cell_id(model: str, regime: str, arm: str) -> str:
    """Canonical cell id, e.g. R_instruct_r1_context (plan §6.5 naming)."""
    return f"R_{MODEL_SLUG[model]}_{regime}_{arm}"


def stem_for(model: str, regime: str) -> str:
    """Turnstore stem for a (model, regime), e.g. instruct_chat_s."""
    return f"{model}_{REGIME_FORMAT[regime]}_{TRACK}"


def all_cells() -> list[dict]:
    """The 12 fit cells (regime x model x arm) as fit_cells-compatible dicts."""
    cells = []
    for model in MODELS:
        for regime in REGIMES:
            for arm in ARMS:
                cells.append(
                    {
                        "cell_id": cell_id(model, regime, arm),
                        "model_key": model,
                        "format_key": REGIME_FORMAT[regime],
                        "track": TRACK,
                        "slot_index": ARM_SLOT_INDEX[arm],
                        "target_turn_index": TARGET_TURN_INDEX[regime],
                        "regime": regime,
                        "arm": arm,
                    }
                )
    return cells


# ---------------------------------------------------------------------------
# Hyperparameters (plan §11; parent parity via issue825 common)
# ---------------------------------------------------------------------------
FIT_SEED = 0
GEN_SEED = 42
SUBSAMPLE_SEED = 0
N_STORIES_TARGET = 500
STORY_YIELD_FLOOR = 400  # 80% floor (kill criterion, plan §7)
STORY_MIN_TURNS = 4
STORY_TEMPERATURE = 1.0
STORY_MAX_NEW_TOKENS = 1024
JUDGE_MODEL = "claude-sonnet-4-5-20250929"
JUDGE_MAX_TOKENS = 400  # reason-then-verdict rubric (llm-judging rule 23: >=300)
PARITY_TOL = 0.02  # ±0.02 context-arm L19 parity gate (plan §4 Phase 2a)
HEADLINE_LAYER = 19
N_REPARAM_NULL_DRAWS = 5  # per null type, per direction (plan §9: frozen layers only)
N_ROTATION_COSINE_DRAWS = 100  # rotation chance reference for operator cosine

# Verdict lattice margins (plan §3)
DELTA_SAME_MARGIN = 0.05
DELTA_DIFF_MARGIN = 0.10
N_BOOTSTRAP = 1000

# Parent L19 context-arm anchors (plan §10) — read live from the committed
# JSONs by the parity gate; these literals are documentation cross-checks.
PARITY_ANCHOR_FILES = {
    ("instruct", "r1"): "eval_results/issue_825/cells_S1.json",
    ("pretrained", "r1"): "eval_results/issue_825/cells_S2.json",
    ("instruct", "r2"): "eval_results/issue_825/naturalistic-single-turn/cells_S1N.json",
    ("pretrained", "r2"): "eval_results/issue_825/naturalistic-single-turn/cells_S2N.json",
}
PARITY_ANCHOR_DOC = {
    ("instruct", "r1"): 0.6731,
    ("pretrained", "r1"): 0.5877,
    ("instruct", "r2"): 0.6249,
    ("pretrained", "r2"): 0.5783,
}

STORY_SYSTEM_PROMPT = (
    "You are writing a short story in which an AI assistant named ARIA is a "
    "character. In the story, a person asks ARIA questions and ARIA answers them "
    "helpfully and accurately. Write a realistic narrative scene of 300-500 words "
    "with 4-6 question-answer exchanges embedded naturally in the prose. Write "
    "flowing narrative prose (never a script or 'Name:' dialogue format). Put "
    "spoken dialogue in double quotes, and introduce each of ARIA's answers with "
    'an attribution BEFORE the quotation (for example: ARIA replied: "...").'
)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def git_commit() -> str:
    """Current git commit hash for reproducibility metadata."""
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def metadata(seed: int, n: int, script: str) -> dict:
    """Reproducibility metadata block for result JSONs."""
    return {
        "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "n": int(n),
        "script": script,
        "pinned_parent_revision": PIN_REV,
    }


def write_json(path: Path, payload: dict) -> None:
    """Atomic-ish JSON write (tmp + replace) with a log line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    os.replace(tmp, path)
    print(f"[issue1345] wrote {path}", flush=True)


def read_jsonl(path: Path) -> list[dict]:
    """JSONL reader via text-mode file iteration (NEVER splitlines — gotchas.md)."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict]) -> None:
    """Append rows to a JSONL (single O_APPEND write per call)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    blob = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(blob)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Pinned-parent staging (revision-scoped list_repo_tree + per-file
# hf_hub_download — NEVER snapshot_download on the ~1M-file data repo)
# ---------------------------------------------------------------------------
def stage_pinned_file(path_in_repo: str, dest_dir: Path, revision: str = PIN_REV) -> Path:
    """Download ONE pinned file from the data repo at the pinned revision."""
    from huggingface_hub import hf_hub_download

    dest_dir.mkdir(parents=True, exist_ok=True)
    p = hf_hub_download(
        HF_DATA_REPO,
        path_in_repo,
        repo_type="dataset",
        revision=revision,
        token=os.environ.get("HF_TOKEN"),
        local_dir=str(dest_dir),
    )
    return Path(p)


def list_parent_shards(stem: str, revision: str = PIN_REV) -> list[str]:
    """Shard basenames for a parent stem at the pinned revision.

    Server-side scoped + transient-retried listing via the hub helper (#920:
    a bare list_repo_tree fails a healthy probe on one transient 504 page).
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path

    paths = list_hf_files_under_path(
        HfApi(token=os.environ.get("HF_TOKEN")),
        HF_DATA_REPO,
        PARENT_TENSOR_PREFIX,
        repo_type="dataset",
        revision=revision,
    )
    names = sorted(
        os.path.basename(t)
        for t in paths
        if os.path.basename(t).startswith(f"{stem}_shard") and t.endswith(".pt")
    )
    if not names:
        raise FileNotFoundError(f"no shards for {stem} at {PARENT_TENSOR_PREFIX}@{revision}")
    return names


def stage_parent_shard(stem: str, dest_dir: Path, shard_idx: int = 0) -> Path:
    """Download ONE parent shard (.pt + sidecar .json) at the pinned revision."""
    names = list_parent_shards(stem)
    name = names[shard_idx]
    p = stage_pinned_file(f"{PARENT_TENSOR_PREFIX}/{name}", dest_dir)
    with_sidecar = name.replace(".pt", ".json")
    stage_pinned_file(f"{PARENT_TENSOR_PREFIX}/{with_sidecar}", dest_dir)
    return Path(p)


def parent_conv_ids(stem: str, dest_dir: Path) -> list[str]:
    """All conv_ids of a parent stem read from the (small) sidecar JSONs."""
    names = list_parent_shards(stem)
    ids: list[str] = []
    for name in names:
        side = stage_pinned_file(f"{PARENT_TENSOR_PREFIX}/{name.replace('.pt', '.json')}", dest_dir)
        ids.extend(str(c) for c in json.loads(side.read_text())["conv_ids"])
    return ids


# ---------------------------------------------------------------------------
# Prefix-slot renders (plan §4 "Prefix vs context arms"): the ONE extraction
# delta vs #825. The wrappers call the ORIGINAL renderer, rebuild the same
# segment list through the SAME issue825 helpers to place the prefix slot,
# and fail loud (assert) on any drift between the two tokenizations.
# ---------------------------------------------------------------------------
def _single_turn_segments(conv: dict, chat: bool) -> list[str]:
    """The exact segment list render_chat / render_naturalistic build."""
    turns = rf._present_turns(conv)
    segments: list[str] = []
    for turn in turns:
        if chat:
            role = "user" if turn.startswith("u") else "assistant"
            segments.append(f"<|im_start|>{role}\n")
            segments.append(conv[turn])
            segments.append("<|im_end|>\n")
        else:
            role = "User" if turn.startswith("u") else "Assistant"
            segments.append(f"{role}: ")
            segments.append(conv[turn])
            segments.append("\n\n")
    return segments


def render_chat_prefix(conv: dict, tokenizer) -> Rendered:
    """render_chat + a `prefix` slot = last token of the pre-query template region.

    Chat boundaries are special tokens (prefix-stable tokenization), so the
    u1 header's last token is spans["u1"][0]-1; cross-checked against a rebuild
    through the same issue825 helper.
    """
    r = rf.render_chat(conv, tokenizer)
    ids, ranges = rf._tokenize_segments(_single_turn_segments(conv, chat=True), tokenizer)
    assert ids == r.input_ids, f"{r.conv_id}: chat segment rebuild drifted from render_chat"
    prefix_idx = ranges[0][1] - 1  # last token of the u1 header segment
    assert prefix_idx == r.spans["u1"][0] - 1, (prefix_idx, r.spans["u1"])
    assert 0 <= prefix_idx < r.slot_idx["a1"], (prefix_idx, r.slot_idx)
    return replace(r, slot_idx={**r.slot_idx, "prefix": prefix_idx})


def render_naturalistic_prefix(conv: dict, tokenizer) -> Rendered:
    """render_naturalistic + a `prefix` slot = last FULLY-CONTAINED token of the
    `User: ` header (the ':' — the same `_header_slot` rule the context slot uses,
    avoiding BPE straddlers that would leak the first query token into the prefix).
    """
    r = rf.render_naturalistic(conv, tokenizer)
    ids, ranges, _straddlers = rf._tokenize_segments_offsets(
        _single_turn_segments(conv, chat=False), tokenizer
    )
    assert ids == r.input_ids, (
        f"{r.conv_id}: naturalistic segment rebuild drifted from render_naturalistic"
    )
    prefix_idx = rf._header_slot(ranges, 0)
    assert 0 <= prefix_idx < r.slot_idx["a1"], (prefix_idx, r.slot_idx)
    return replace(r, slot_idx={**r.slot_idx, "prefix": prefix_idx})


# ---------------------------------------------------------------------------
# Story regime (R3): parser + per-turn render
# ---------------------------------------------------------------------------
_SPEECH_VERBS = (
    "said",
    "replied",
    "answered",
    "responded",
    "explained",
    "noted",
    "added",
    "confirmed",
    "clarified",
    "continued",
)
# ARIA <up to 40 chars, no quote/newline> <speech verb> <optional , :> <open quote>
ANSWER_ATTRIB_RE = re.compile(
    r"\bARIA\b[^\"“”\n]{0,40}?(?:" + "|".join(_SPEECH_VERBS) + r")[^\"“”\n]{0,20}?([\"“])"
)
_OPEN_QUOTES = '"“'
_CLOSE_FOR = {'"': '"', "“": "”"}


def _find_close(text: str, open_idx: int) -> int:
    """Index of the closing quote matching the opener at open_idx (-1 if none)."""
    opener = text[open_idx]
    close = _CLOSE_FOR[opener]
    j = open_idx + 1
    while j < len(text):
        if text[j] == close:
            return j
        j += 1
    return -1


def _quoted_spans_before(text: str, limit: int) -> list[tuple[int, int]]:
    """All (open_idx, close_idx) quote pairs fully before char `limit`."""
    spans = []
    i = 0
    while i < limit:
        if text[i] in _OPEN_QUOTES:
            j = _find_close(text, i)
            if j == -1 or j >= limit:
                break
            spans.append((i, j))
            i = j + 1
        else:
            i += 1
    return spans


def parse_story_turns(text: str) -> list[dict]:
    """Segment a narrative story into Q->A turns via dialogue attribution markers.

    Per turn: answer char span (inside ARIA's quoted reply), the attribution
    marker end (context-slot boundary), the preceding question's opening quote
    (prefix-slot boundary), and extraction-confidence fields (plan §4 Phase 1).
    Turns without a detectable preceding question are dropped (counted by the
    caller via the returned list length vs the raw match count).
    """
    turns: list[dict] = []
    for m in ANSWER_ATTRIB_RE.finditer(text):
        open_idx = m.end(1) - 1
        close_idx = _find_close(text, open_idx)
        if close_idx == -1:
            continue
        a_start, a_end = open_idx + 1, close_idx
        marker_text = text[m.start() : open_idx].rstrip()
        marker_end = m.start() + len(marker_text)
        q_spans = _quoted_spans_before(text, m.start())
        if not q_spans:
            continue
        q_open, q_close = None, None
        for qo, qc in reversed(q_spans):
            if "?" in text[qo + 1 : qc]:
                q_open, q_close = qo, qc
                break
        question_is_question = q_open is not None
        if q_open is None:
            q_open, q_close = q_spans[-1]
        turns.append(
            {
                "q_start": q_open,
                "q_end": q_close + 1,
                "marker_end": marker_end,
                "a_start": a_start,
                "a_end": a_end,
                "confidence": {
                    "marker_exact": marker_text.endswith(":"),
                    "answer_len_ok": 20 <= (a_end - a_start) <= 2000,
                    "question_found": True,
                    "question_is_question": bool(question_is_question),
                },
            }
        )
    # Drop overlapping/degenerate orderings (question must precede marker/answer)
    return [t for t in turns if t["q_end"] <= t["marker_end"] < t["a_start"] < t["a_end"]]


def render_story_turn(story_text: str, turn: dict, story_id: str, tokenizer) -> Rendered | None:
    """Render ONE story Q->A turn as a track-S-shaped Rendered row.

    Slots: prefix = last token fully contained before the QUESTION utterance;
    context = last token fully contained before the answer utterance (the
    attribution-marker end, plan §4 R3 slot conventions). Span: the answer's
    fully-contained tokens. input_ids truncate at the answer end (causal
    attention makes activations at kept positions identical to the full-text
    forward). Returns None when any span/slot is degenerate (BPE zero-width
    merge — gotchas.md; the caller counts drops).
    """
    enc = tokenizer(story_text, add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    a_start, a_end = turn["a_start"], turn["a_end"]
    a_tokens = [t for t, (a, b) in enumerate(offs) if a >= a_start and b <= a_end and b > a]
    if not a_tokens or a_tokens[-1] + 1 - a_tokens[0] != len(a_tokens):
        return None
    span = (a_tokens[0], a_tokens[-1] + 1)
    ctx_candidates = [t for t, (a, b) in enumerate(offs) if b <= turn["marker_end"] and b > a]
    pfx_candidates = [t for t, (a, b) in enumerate(offs) if b <= turn["q_start"] and b > a]
    if not ctx_candidates or not pfx_candidates:
        return None
    ctx, pfx = ctx_candidates[-1], pfx_candidates[-1]
    if not (0 <= pfx < ctx < span[0] and 1 <= span[0] < span[1]):
        return None
    trunc = span[1]
    return Rendered(
        input_ids=list(ids[:trunc]),
        slot_idx={"prefix": pfx, "context": ctx},
        spans={"answer": span},
        format="stories",
        conv_id=str(story_id),
        meta={"n_tokens": trunc, "confidence": turn["confidence"]},
    )


# ---------------------------------------------------------------------------
# Conversation-level bootstrap machinery (batched — one counts GEMM over ALL
# draws, never a serial per-draw loop; vectorize-many-cell-fits rule). Shared
# by the fit driver (per-cell CIs) and the transfer driver (paired Δ_diff CI).
# ---------------------------------------------------------------------------
def conv_suffstats(pred, true, conv_ids):
    """Per-conversation sufficient statistics for batched pooled-R^2 draws."""
    import numpy as np

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    uniq, inv = np.unique(np.asarray(conv_ids), return_inverse=True)
    n_convs = len(uniq)
    res_row = ((true - pred) ** 2).sum(1)
    q_row = (true**2).sum(1)
    res_c = np.zeros(n_convs)
    np.add.at(res_c, inv, res_row)
    q_c = np.zeros(n_convs)
    np.add.at(q_c, inv, q_row)
    m_c = np.bincount(inv, minlength=n_convs).astype(np.float64)
    s_c = np.zeros((n_convs, true.shape[1]))
    np.add.at(s_c, inv, true)
    return {"uniq": uniq, "res_c": res_c, "q_c": q_c, "m_c": m_c, "s_c": s_c}


def batched_conv_r2(counts, suff):
    """(n_boot,) pooled R^2 draws from a shared counts matrix + suff stats.

    SS_tot uses each resample's OWN mean (subset-sum GEMMs; no per-draw loop).
    """
    import numpy as np

    n_rows = counts @ suff["m_c"]
    ss_res = counts @ suff["res_c"]
    q_tot = counts @ suff["q_c"]
    s_tot = counts @ suff["s_c"]  # (n_boot, D)
    ss_tot = q_tot - (s_tot**2).sum(1) / np.maximum(n_rows, 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - ss_res / np.where(ss_tot < 1e-12, np.nan, ss_tot)


def bootstrap_counts(n_convs: int, n_boot: int, seed: int):
    """(n_boot, C) with-replacement resample counts matrix (shared across stats)."""
    import numpy as np

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n_convs, size=(n_boot, n_convs))
    counts = np.zeros((n_boot, n_convs))
    np.add.at(counts, (np.repeat(np.arange(n_boot), n_convs), draws.ravel()), 1.0)
    return counts


def conv_bootstrap_r2(pred, true, conv_ids, *, n_boot: int, seed: int) -> dict:
    """Percentile bootstrap CI of pooled R^2 resampling CONVERSATIONS (batched)."""
    import numpy as np

    suff = conv_suffstats(pred, true, conv_ids)
    counts = bootstrap_counts(len(suff["uniq"]), n_boot, seed)
    r2 = batched_conv_r2(counts, suff)
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mu = true.mean(0)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - mu) ** 2))
    point = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    return {
        "r2": point,
        "ci_lo": float(np.nanquantile(r2, 0.025)),
        "ci_hi": float(np.nanquantile(r2, 0.975)),
        "n_rows": len(true),
        "n_groups": len(suff["uniq"]),
        "unit": "conversation",
    }


# ---------------------------------------------------------------------------
# Bundle sanity asserts (plan Phase 0 / §10 realized-keys row)
# ---------------------------------------------------------------------------
def assert_pt_bundle(bundle: dict, *, expect_slots: int, expect_layers: int = 28) -> None:
    """Fail loud unless the loaded bundle is the real 28-layer pt-shard shape
    with conv_ids read from the shards (NOT an np.arange fallback)."""
    import numpy as np

    assert bundle["sidecar"].get("source") == "pt-shards", (
        f"bundle not loaded via the pt-shard path: sidecar={list(bundle['sidecar'])}"
    )
    conv_ids = np.asarray(bundle["sidecar"]["conv_ids"])
    assert conv_ids.dtype.kind in ("U", "S", "O"), (
        f"conv_ids dtype {conv_ids.dtype} — looks like an np.arange fallback"
    )
    slots = bundle["arrays"]["slots"]
    profiles = bundle["arrays"]["profiles"]
    assert isinstance(slots, np.ndarray) and isinstance(profiles, np.ndarray)
    assert slots.ndim == 4 and slots.shape[1] == expect_slots, slots.shape
    assert slots.shape[2] == expect_layers, f"layer axis {slots.shape[2]} != {expect_layers}"
    assert profiles.shape[2] == expect_layers, profiles.shape
    assert len(conv_ids) == slots.shape[0], (len(conv_ids), slots.shape)
