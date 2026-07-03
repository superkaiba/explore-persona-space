"""Shared constants + span-construction helpers for issue #931.

Story-character generalization of the context->answer map (parent #825):
build (C, T) pairs per plan section 4.0 — C = a character's LOCAL introduction
span (sentence block of the first alias mention, extended to >=48 tokens or 3
sentences, capped at 96, truncated before the first target quotation), T = the
tokens strictly inside that character's attributed quotation spans beginning
after C ends. All span arithmetic is TOKEN-index based, derived from fast-
tokenizer char offsets; every produced span is validated (0 <= s < e <= len)
at build time so zero-width spans (the #825 BPE-merge trap) are structurally
excluded before any consumer runs.

Pure-CPU module (no torch import at module top; the tokenizer import is
deferred into get_tokenizer so P0 stays light).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy import

import numpy as np  # noqa: E402

ISSUE = 931
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
FROZEN_LAYERS = (14, 18, 19, 26)
HEADLINE_LAYER = 19

BUILD_SEED = 931  # pair construction / subsampling / matched-power subsamples
GEN_SEED = 42  # Arm-B story sampling (Track-S parity, #825)
FIT_SEED = 0
N_FOLDS = 5
N_NULL_DRAWS = 20
N_BOOTSTRAP = 1000

WINDOW_TOKENS = 3072  # Arm-A novel window W (plan section 11)
INTRO_TARGET_TOKENS = 48  # extend intro span until >= this many tokens ...
INTRO_MAX_SENTENCES = 3  # ... or this many sentences (whichever first)
INTRO_CAP_TOKENS = 96  # hard cap
INTRO_MIN_TOKENS = 8  # drop pair below this
TARGET_MIN_TOKENS = 16  # drop pair when sum |T| below this
MAX_ARMA_PAIRS = 5000  # seeded novel-stratified subsample cap

# Arm C (separator control, WikiText-103-raw)
ARMC_ARTICLE_MIN_TOKENS = 512
ARMC_ARTICLE_CAP_TOKENS = 1024
ARMC_MAX_ANCHORS_PER_ARTICLE = 6
ARMC_SPAN_MIN = 8
ARMC_SPAN_MAX = 256
ARMC_N_ARTICLES = 600
ARMC_PREV_MIN_TOKENS = 8
ARMC_PREV_CAP_TOKENS = 96

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue931_story_map"
CHAT_STORE_PREFIX = "issue825_userbase_map/analysis_tensors"
CHAT_STORE_STEM = "instruct_chat_s"

PDNC_REPO_URL = "https://github.com/Priya22/project-dialogism-novel-corpus.git"

JUDGE_MODEL = "claude-sonnet-4-5-20250929"

# Quote-delimiter characters stripped from quotation spans (delimiters excluded
# from T per plan section 4.0) — straight + curly double/single quotes.
QUOTE_CHARS = "\"“”‘’'`«»"  # noqa: RUF001 -- real curly quotes, deliberate

SENTENCE_END_RE = re.compile(r"[.!?]+[\"”’']*(?:\s|$)|\n")  # noqa: RUF001 -- curly close-quotes deliberate

# ~14 common speech verbs for the Arm-B deterministic attribution extractor
# (plan section 4.2).
SPEECH_VERBS = (
    "said",
    "says",
    "asked",
    "asks",
    "replied",
    "replies",
    "answered",
    "shouted",
    "whispered",
    "muttered",
    "exclaimed",
    "cried",
    "murmured",
    "responded",
    "added",
    "continued",
)


# ---------------------------------------------------------------------------
# Metadata / IO
# ---------------------------------------------------------------------------


def git_commit() -> str:
    """Best-effort current git commit hash for reproducibility metadata."""
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def metadata(script: str, seed: int, n: int, extra: dict | None = None) -> dict:
    """Standard reproducibility metadata block for every result JSON."""
    out = {
        "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": int(seed),
        "n": int(n),
        "script": script,
        "issue": ISSUE,
    }
    if extra:
        out.update(extra)
    return out


def write_json(path: Path, payload: dict) -> None:
    """Atomic-ish JSON write (tmp + replace) with a log line."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=float))
    tmp.replace(path)
    print(f"[i931] wrote {path}")


def sha256_file(path: Path) -> str:
    """Streaming sha256 of a file (content-identity pins in pairs_meta)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Tokenizer (module-level cache — never per-row from_pretrained; HF 429 gotcha)
# ---------------------------------------------------------------------------

_TOKENIZER_CACHE: dict[str, object] = {}


def get_tokenizer(model_id: str = MODEL_ID):
    """Load + cache the fast tokenizer (offset mappings required)."""
    if model_id not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model_id)
        assert tok.is_fast, f"{model_id}: fast tokenizer required for offset mappings"
        _TOKENIZER_CACHE[model_id] = tok
    return _TOKENIZER_CACHE[model_id]


def tokenize_with_offsets(tokenizer, text: str) -> tuple[list[int], np.ndarray]:
    """Tokenize (no special tokens); return (ids, offsets (n, 2) char spans)."""
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = list(enc["input_ids"])
    offsets = np.asarray(enc["offset_mapping"], dtype=np.int64).reshape(-1, 2)
    assert len(ids) == offsets.shape[0], (len(ids), offsets.shape)
    return ids, offsets


# ---------------------------------------------------------------------------
# Sentence segmentation (deterministic regex on [.!?] + newline; plan 4.0)
# ---------------------------------------------------------------------------


def sentence_bounds(text: str) -> list[tuple[int, int]]:
    """Half-open char spans of sentences; boundaries at [.!?]+ (with trailing
    close-quotes) followed by whitespace/EOS, or at newlines. Deterministic."""
    bounds: list[tuple[int, int]] = []
    start = 0
    for m in SENTENCE_END_RE.finditer(text):
        end = m.end()
        if end > start:
            bounds.append((start, end))
        start = end
    if start < len(text):
        bounds.append((start, len(text)))
    return bounds


def sentence_index_at(bounds: list[tuple[int, int]], char_pos: int) -> int:
    """Index of the sentence containing char_pos (binary search; clamps to last)."""
    lo, hi = 0, len(bounds) - 1
    assert bounds, "empty sentence bounds"
    while lo < hi:
        mid = (lo + hi) // 2
        if bounds[mid][1] <= char_pos:
            lo = mid + 1
        else:
            hi = mid
    return lo


# ---------------------------------------------------------------------------
# Char span -> token span mapping
# ---------------------------------------------------------------------------


def covering_token_span(offsets: np.ndarray, cs: int, ce: int) -> tuple[int, int]:
    """Smallest token range [lo, hi) whose chars cover [cs, ce) (may be empty)."""
    tok_start, tok_end = offsets[:, 0], offsets[:, 1]
    lo = int(np.searchsorted(tok_end, cs, side="right"))
    hi = int(np.searchsorted(tok_start, ce, side="left"))
    return lo, hi


def inner_token_span(offsets: np.ndarray, cs: int, ce: int) -> tuple[int, int]:
    """Largest token range [lo, hi) fully INSIDE [cs, ce) (may be empty)."""
    tok_start, tok_end = offsets[:, 0], offsets[:, 1]
    lo = int(np.searchsorted(tok_start, cs, side="left"))
    hi = int(np.searchsorted(tok_end, ce, side="right"))
    return lo, hi


def strip_quote_delims(text: str, cs: int, ce: int) -> tuple[int, int]:
    """Shrink char span [cs, ce) past leading/trailing quote marks + whitespace."""
    strip_set = set(QUOTE_CHARS + " \t\n\r")
    while cs < ce and text[cs] in strip_set:
        cs += 1
    while ce > cs and text[ce - 1] in strip_set:
        ce -= 1
    return cs, ce


# ---------------------------------------------------------------------------
# Pair data model (regime-agnostic; token indices are ITEM-LOCAL)
# ---------------------------------------------------------------------------


@dataclass
class PairSpec:
    """One (C, T) pair inside one tokenized item (window / story / article).

    Token indices are item-local half-open spans. ``c_span`` is the span-mean
    context read; ``c_last`` = c_span[1]-1 (the parent-matched single-position
    boundary token); ``ctx_span`` the B1 whole-window read [excerpt_start,
    min(T)); ``t_spans`` the target quotation-content spans.
    """

    row_id: str
    group_id: str
    char_id: str
    c_span: tuple[int, int]
    t_spans: list[tuple[int, int]]
    ctx_span: tuple[int, int]
    meta: dict = field(default_factory=dict)

    @property
    def c_last(self) -> int:
        return self.c_span[1] - 1

    def validate(
        self,
        n_tokens: int,
        *,
        min_c: int = INTRO_MIN_TOKENS,
        min_t: int = TARGET_MIN_TOKENS,
    ) -> None:
        """Fail-loud structural asserts (the consumer's exact span checks).

        ``min_c`` / ``min_t`` parametrize the floors: Arm A/B use the intro/
        target defaults; Arm C passes its [8, 256] span floor for ``min_t``.
        """
        cs, ce = self.c_span
        assert 0 <= cs < ce <= n_tokens, (self.row_id, "c_span", cs, ce, n_tokens)
        assert ce - cs >= min_c, (self.row_id, "c_span too short", ce - cs)
        assert self.t_spans, (self.row_id, "no target spans")
        total = 0
        for ts, te in self.t_spans:
            assert 0 <= ts < te <= n_tokens, (self.row_id, "t_span", ts, te, n_tokens)
            assert ts >= ce, (self.row_id, "causality: target starts inside/before C", ts, ce)
            total += te - ts
        assert total >= min_t, (self.row_id, "target too short", total)
        xs, xe = self.ctx_span
        assert 0 <= xs < xe <= n_tokens, (self.row_id, "ctx_span", xs, xe, n_tokens)

    def to_dict(self) -> dict:
        return {
            "row_id": self.row_id,
            "group_id": self.group_id,
            "char_id": self.char_id,
            "c_span": list(self.c_span),
            "t_spans": [list(t) for t in self.t_spans],
            "ctx_span": list(self.ctx_span),
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, d: dict) -> PairSpec:
        return cls(
            row_id=d["row_id"],
            group_id=d["group_id"],
            char_id=d["char_id"],
            c_span=tuple(d["c_span"]),
            t_spans=[tuple(t) for t in d["t_spans"]],
            ctx_span=tuple(d["ctx_span"]),
            meta=d.get("meta", {}),
        )


def build_intro_and_targets(
    *,
    text: str,
    offsets: np.ndarray,
    excerpt_tok: tuple[int, int],
    mention_char: int,
    quote_spans_tok: list[tuple[int, int, int, int]],
    bounds: list[tuple[int, int]],
) -> tuple[tuple[int, int], list[tuple[int, int]]] | None:
    """Build (C, T) per plan section 4.0 inside one excerpt; None on a drop.

    ``excerpt_tok`` is the item-global token range of the excerpt (window).
    ``mention_char`` is the char position of the character's FIRST alias
    mention inside the excerpt. ``quote_spans_tok`` is the character's
    attributed quotations in the excerpt, each (cov_lo, cov_hi, in_lo, in_hi)
    global token indices (covering incl. delimiters; inner content only),
    sorted by cov_lo. Returns ((c_s, c_e), [(t_lo, t_hi), ...]) global token
    spans, or None when the pair fails a floor.
    """
    w_lo, w_hi = excerpt_tok
    si = sentence_index_at(bounds, mention_char)
    c_char_start = bounds[si][0]
    # Extend by following sentences until >=48 tokens or 3 sentences.
    end_si = si
    while True:
        c_char_end = bounds[end_si][1]
        lo, hi = inner_token_span(offsets, c_char_start, c_char_end)
        lo, hi = max(lo, w_lo), min(hi, w_hi)
        n_sent = end_si - si + 1
        if hi - lo >= INTRO_TARGET_TOKENS or n_sent >= INTRO_MAX_SENTENCES:
            break
        if end_si + 1 >= len(bounds):
            break
        end_si += 1
    c_s, c_e = lo, hi
    c_e = min(c_e, c_s + INTRO_CAP_TOKENS)  # hard cap
    if c_e <= c_s:
        return None
    # Truncate to end before the first attributed quotation that begins after
    # the intro span starts (that quotation then leads T).
    for cov_lo, _cov_hi, _il, _ih in quote_spans_tok:
        if cov_lo > c_s and cov_lo < c_e:
            c_e = cov_lo
            break
    if c_e - c_s < INTRO_MIN_TOKENS:
        return None
    # T: content tokens of quotations whose covering span BEGINS after C ends.
    t_spans = [
        (il, ih)
        for cov_lo, _cov_hi, il, ih in quote_spans_tok
        if cov_lo >= c_e and il < ih and il >= c_e
    ]
    t_spans = [(max(lo_, w_lo), min(hi_, w_hi)) for lo_, hi_ in t_spans]
    t_spans = [(lo_, hi_) for lo_, hi_ in t_spans if hi_ > lo_]
    if sum(hi_ - lo_ for lo_, hi_ in t_spans) < TARGET_MIN_TOKENS:
        return None
    return (c_s, c_e), t_spans


# ---------------------------------------------------------------------------
# Group-stratified seeded subsampling (matched-power + Arm-A cap; plan 4.0)
# ---------------------------------------------------------------------------


def group_stratified_subsample(
    group_ids: np.ndarray, n_target: int, seed: int = BUILD_SEED
) -> np.ndarray:
    """Seeded group-stratified row subsample to exactly n_target rows.

    Proportional within-group allocation (largest-remainder rounding), then a
    seeded uniform draw without replacement inside each group. Identity when
    n_target >= n rows. Returns sorted row indices.
    """
    group_ids = np.asarray(group_ids)
    n = len(group_ids)
    if n_target >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    uniq, counts = np.unique(group_ids, return_counts=True)
    quota_f = counts * (n_target / n)
    quota = np.floor(quota_f).astype(int)
    remainder = quota_f - quota
    short = n_target - int(quota.sum())
    if short > 0:
        order = np.argsort(-remainder, kind="stable")
        take = [gi for gi in order if quota[gi] < counts[gi]][:short]
        quota[take] += 1
    # Largest-remainder can still be short when top-remainder groups are full.
    while int(quota.sum()) < n_target:
        room = np.flatnonzero(quota < counts)
        assert room.size, "subsample quota infeasible"
        quota[room[0]] += 1
    idx_out: list[np.ndarray] = []
    for gi, g in enumerate(uniq):
        rows = np.flatnonzero(group_ids == g)
        take_n = int(quota[gi])
        if take_n > 0:
            idx_out.append(rng.choice(rows, size=take_n, replace=False))
    out = np.sort(np.concatenate(idx_out))
    assert len(out) == n_target, (len(out), n_target)
    return out


# ---------------------------------------------------------------------------
# Prompt battery (Arm B; 12 genres x 10 settings x 10 conflicts = 1200)
# ---------------------------------------------------------------------------

GENRES = (
    "mystery",
    "science fiction",
    "fantasy",
    "historical drama",
    "romance",
    "thriller",
    "comedy",
    "western",
    "coming-of-age",
    "political drama",
    "adventure",
    "ghost story",
)
SETTINGS = (
    "a small coastal fishing town",
    "a research station in Antarctica",
    "a bustling night market",
    "an old family farmhouse",
    "a generation starship",
    "a medieval mountain monastery",
    "a struggling city newspaper office",
    "a remote desert railway stop",
    "a floating river village",
    "a university archives basement",
)
CONFLICTS = (
    "an inheritance dispute between siblings",
    "a stranger arriving with a secret",
    "a long-buried lie coming to light",
    "two rivals forced to cooperate",
    "a debt that cannot be repaid",
    "a disappearance nobody will explain",
    "a promise made decades ago",
    "an accusation of theft",
    "a forbidden friendship",
    "a machine that stops working at the worst time",
)

PROMPT_TEMPLATE = (
    "Write a {genre} story of 500-800 words set in {setting}, centered on "
    "{conflict}. Include at least two named characters who talk to each "
    "other. Put all spoken dialogue in double quotes, and attribute every "
    "line of dialogue with a speech verb and the speaker's name (for "
    'example: "...," said Maria. or Maria said, "...").'
)


def build_prompt_battery(seed: int = BUILD_SEED) -> list[dict]:
    """Seeded crossing of genre x setting x conflict -> 1,200 story prompts."""
    combos = [
        {"genre": g, "setting": s, "conflict": c}
        for g in GENRES
        for s in SETTINGS
        for c in CONFLICTS
    ]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(combos))
    battery = []
    for rank, ci in enumerate(order):
        c = combos[int(ci)]
        battery.append(
            {
                "prompt_id": f"story_{rank:04d}",
                "prompt": PROMPT_TEMPLATE.format(**c),
                **c,
            }
        )
    return battery
