"""Issue #744 — shared constants + small helpers for the continuity rig.

Centralises the corpus identities, HF upload target, model defaults, and the
clause-opener closed-class wordlist so the four issue744 scripts agree on every
pinned value (the reproducibility-card source of truth, plan §10).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── HF upload target (plan §10) ────────────────────────────────────────────────
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_OVERFLOW_REPO = "superkaiba1/explore-persona-space-overflow"
HF_PREFIX = "issue744_token_continuity"

# ── Model defaults (plan §10) ──────────────────────────────────────────────────
DEFAULT_MODEL = "Qwen/Qwen2.5-7B"
INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584

# ── Corpus A — Natural Stories (plan §4.2) ─────────────────────────────────────
NS_REPO = "languageMIT/naturalstories"
NS_BRANCH = "master"
NS_STORIES_FILE = "naturalstories_RTS/all_stories.tok"
NS_PARSES_FILE = "parses/penn/all-parses.txt.penn"
NS_EXPECTED_WORDS = 10256
NS_EXPECTED_ITEMS = 10

# ── Corpus B — WikiText-103 raw train split (plan §4.2) ────────────────────────
WIKITEXT_REPO = "Salesforce/wikitext"
WIKITEXT_CONFIG = "wikitext-103-raw-v1"
WIKITEXT_REVISION = "b08601e04326"
WIKITEXT_SPLIT = "train"
BROADER_SEED = 744
BROADER_N_SEQUENCES = 2000
BROADER_TOKEN_BUDGET = 1_000_000
MIN_CHARS = 32

# ── Hyperparameters (plan §10) ─────────────────────────────────────────────────
SEED = 744
TRAJECTORY_WINDOW_K = 3
DIRECTION_PRES_STEPS = (0, 1, 2, 3)
ROGUE_DIM_TOPK = 3
MAX_SEQ_LEN = 1024
# Natural Stories overlapping-chunk stride (Barenholtz 2606.05346 §2.1, #744 C2).
# Stories exceed the 1024-token context window, so each story is processed in
# overlapping MAX_SEQ_LEN-token chunks at this stride (512 = 50% overlap) and the
# per-position reads are de-duplicated to cover the FULL story (no truncation).
# 50% overlap >> the k=3 fit + max(+3) lookahead context floor, so every
# de-duplicated position has its OLS-fit window and lookahead fully in-chunk.
NS_CHUNK_STRIDE = 512
RANDOM_BASELINE_N_PAIRS = 100_000
# Broader random-baseline reservoir pool size (#744 random-pair-memory concern).
# The broader corpus is STREAMED (no full raw retention), so its random baseline
# is reservoir-sampled over the FULL Phase-1 stream into a fixed per-layer pool of
# this many raw token vectors; the per-flavor abs-cosine over RANDOM_BASELINE_N_PAIRS
# pairs drawn from the pool is then computed once and stored as
# ``broader_random_pairs.pt`` (the analyzer reads that, never re-concatenates the
# bounded broader_raw subset). 20k fp16 vectors/layer x 28 layers x 3584 ~ 4 GB,
# well under the 50 GB VM analysis floor; >> the 20k pairs the closed-form
# convergence test needs.
BROADER_RANDOM_POOL = 20_000
BOOTSTRAP_B = 2000
# Sun 2402.17762 §2 sink/outlier mask threshold.
SINK_ABS_FLOOR = 100.0
SINK_MEDIAN_RATIO = 1000.0
# Flavor labels (plan §5 conditions table).
FLAVORS = ("raw", "std", "ablate")

# ── Closed-class clause-opener wordlist (plan §4.3 / §11 item 12) ──────────────
# Barenholtz §4.2 enrichment class ("and", "as", "that", "had") + the Penn CC/IN
# closed class of coordinators + complementizers + subordinators. This wordlist
# is the PRIMARY syntactic-boundary mask for the BROADER corpus ONLY (WikiText
# has no gold parses — plan §11 syntactic_mask_broader, flagged as the coarser
# proxy). For Natural Stories the PRIMARY mask is the GOLD Penn clause-opener
# label (first terminal under S/SBAR OR CC/IN — plan §11 syntactic_mask_ns),
# built by ``explore_persona_space.analysis.penn_parser`` and aligned to the NS
# word stream at dump time; this wordlist is emitted alongside it on NS only as
# the A11 gold-vs-proxy cross-check companion, never as the NS primary mask.
CLAUSE_OPENER_WORDS = frozenset(
    {
        # coordinating conjunctions (CC)
        "and",
        "or",
        "but",
        "nor",
        "yet",
        "so",
        "for",
        # complementizers / subordinators (IN)
        "that",
        "if",
        "as",
        "because",
        "while",
        "when",
        "whenever",
        "although",
        "though",
        "since",
        "unless",
        "until",
        "before",
        "after",
        "whereas",
        "whether",
        "where",
        "wherever",
        "than",
        # relative / wh-openers
        "which",
        "who",
        "whom",
        "whose",
        "what",
    }
)


def sha256_file(path: Path) -> str:
    """SHA-256 of a file's bytes (manifest provenance)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    """SHA-256 of an in-memory byte string."""
    return hashlib.sha256(data).hexdigest()


def write_json(path: Path, obj: dict) -> None:
    """Write a JSON object with a stable key order + trailing newline."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def is_clause_opener(word: str) -> bool:
    """Closed-class clause-opener test (case-folded, punctuation-stripped)."""
    w = word.strip().strip(".,;:!?\"'()[]{}").casefold()
    return w in CLAUSE_OPENER_WORDS
