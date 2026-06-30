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
RANDOM_BASELINE_N_PAIRS = 100_000
BOOTSTRAP_B = 2000
# Sun 2402.17762 §2 sink/outlier mask threshold.
SINK_ABS_FLOOR = 100.0
SINK_MEDIAN_RATIO = 1000.0
# Flavor labels (plan §5 conditions table).
FLAVORS = ("raw", "std", "ablate")

# ── Closed-class clause-opener wordlist (plan §4.3 / §11 item 12) ──────────────
# Barenholtz §4.2 enrichment class ("and", "as", "that", "had") + the Penn CC/IN
# closed class of coordinators + complementizers + subordinators. Used as the
# PRIMARY deterministic syntactic-boundary mask for BOTH corpora (the gold Penn
# parse on NS is cross-checked against this proxy — agreement reported, A11).
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
