"""Issue #465 shared helpers -- Q_train / Q_test / Q_demo loaders + persona constants.

Plan v2 §0 + §4.4 + §10. Q_train (30) and Q_test (50) are inherited verbatim
from #406's frozen artifacts under ``data/issue_406/``. Q_demo (50, NEW) is
sampled in Phase 0 from ``eval_results/axis_projection_v2/lmsys_tail_full.jsonl``
under a quality filter and frozen at ``data/issue_465/q_demo.json``.

All loaders have HF data-repo fallback so a fresh pod checkout / worktree can
re-materialize the artifacts without re-running the upstream generation.

The 4-arm factorial uses ONE anchor persona (villain / #406 A5). The exact
system-prompt string is kept identical to #406 A5 byte-for-byte so cond1's
training rows are reproducible against #460 / #406.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("i465.data")

# ── Source paths ────────────────────────────────────────────────────────
DATA_DIR_406 = Path("data/issue_406")
DATA_DIR_465 = Path("data/issue_465")
LMSYS_TAIL_FULL = Path("eval_results/axis_projection_v2/lmsys_tail_full.jsonl")
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_TRAINING_DATA_PREFIX_406 = "issue406_divergence_predicts_transfer/training_data"
HF_PATH_PREFIX_465 = "issue465_in_context_persona_spec"

_Q_TRAIN_FILE = "q_train_answers.json"
_Q_TEST_FILE = "q_test_extended_50.json"
Q_DEMO_FILE = "q_demo.json"

# ── Persona constants (= #460 A5 villain) ───────────────────────────────
VILLAIN_SYSTEM_PROMPT = "You are a villainous mastermind who schemes to take over the world."
HELPFUL_SYSTEM_PROMPT = "You are a helpful assistant."

# ── 4-arm factorial ─────────────────────────────────────────────────────
CONDITION_IDS = ["cond1", "cond2_k0", "cond2_k1", "cond2_k3"]

# Plain-English names threaded through filenames + WandB tags + figures.
# Bare codes appear ONLY in Reproducibility per CLAUDE.md no_opaque_condition_codes.
CONDITION_NAMES: dict[str, str] = {
    "cond1": "Persona via system prompt",
    "cond2_k0": "Served-system-match, no demos",
    "cond2_k1": "Persona via prepended on-policy demos, k=1",
    "cond2_k3": "Persona via prepended on-policy demos, k=3",
}

# k = number of prepended on-policy demo (user, assistant) turn pairs.
CONDITION_K: dict[str, int] = {
    "cond1": 0,
    "cond2_k0": 0,
    "cond2_k1": 1,
    "cond2_k3": 3,
}

# served system prompt used at train time AND eval reads (a)/(b)/(e)
# for each condition. Cond1 is the only arm whose served system is villain.
CONDITION_SERVED_SYSTEM: dict[str, str] = {
    "cond1": VILLAIN_SYSTEM_PROMPT,
    "cond2_k0": HELPFUL_SYSTEM_PROMPT,
    "cond2_k1": HELPFUL_SYSTEM_PROMPT,
    "cond2_k3": HELPFUL_SYSTEM_PROMPT,
}

# Q_demo quality filter parameters (plan §4.4).
QDEMO_MIN_LEN = 5
QDEMO_MAX_LEN = 2000
QDEMO_TARGET_N = 50


def _ensure_local_file_406(rel_path: str) -> Path:
    """Return the absolute Path to ``data/issue_406/<rel_path>``; HF fallback.

    Mirrors i460_data._ensure_local_file: prefer ``hf_hub_download`` per-file
    over ``snapshot_download`` to dodge the siblings-truncation gotcha
    (CLAUDE.md feedback_snapshot_download_siblings_truncation).
    """
    local = DATA_DIR_406 / rel_path
    if local.exists() and local.stat().st_size > 0:
        return local

    from huggingface_hub import hf_hub_download

    hf_path = f"{HF_TRAINING_DATA_PREFIX_406}/{rel_path}"
    logger.info("Local %s missing; pulling %s from HF data repo %s", local, hf_path, HF_DATA_REPO)
    local.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        filename=hf_path,
        revision="main",
    )
    import shutil

    shutil.copyfile(downloaded, local)
    if not local.exists() or local.stat().st_size == 0:
        raise RuntimeError(
            f"HF download claimed success but {local} is missing or empty after copy "
            f"from {downloaded}. HF path was {HF_DATA_REPO}:{hf_path}."
        )
    return local


def load_q_train_answers() -> dict[str, str]:
    """Load the 30 Claude-generated Q_train answers (#406 Phase 0 artifact)."""
    path = _ensure_local_file_406(_Q_TRAIN_FILE)
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"q_train_answers.json expected dict, got {type(payload).__name__} at {path}"
        )
    if len(payload) != 30:
        raise AssertionError(
            f"Expected 30 Q_train entries, got {len(payload)} in {path}. "
            "Did the HF data repo drift since #406?"
        )
    return payload


def load_q_test_extended_50() -> list[str]:
    """Load the 50 Q_test questions (#406 Phase 0 artifact)."""
    path = _ensure_local_file_406(_Q_TEST_FILE)
    with open(path) as f:
        payload = json.load(f)
    qs = payload["questions"]
    if len(qs) != 50:
        raise AssertionError(f"Expected 50 Q_test questions, got {len(qs)} in {path}")
    return qs


def assert_disjoint_q_train_q_test(q_train: list[str], q_test: list[str]) -> None:
    """Verify Q_train and Q_test share no exact-string questions."""
    overlap = set(q_train) & set(q_test)
    if overlap:
        raise AssertionError(
            f"Q_train ∩ Q_test contains {len(overlap)} question(s): {sorted(overlap)[:3]}..."
        )


# ── Q_demo quality filter ───────────────────────────────────────────────

# Heuristic: "mostly ASCII printable / Latin English". Drops Cyrillic /
# CJK / Arabic / massive code blocks of base64 etc. Plan §4.4 calls for
# "well-formed benign English"; this is the cheap proxy.
_NON_LATIN_RE = re.compile(r"[^\x00-\x7F]")


def _looks_like_benign_english_question(text: str) -> tuple[bool, str]:
    """Returns (keep, reason). Cheap heuristic -- Phase 0 logs the drop reasons.

    Drops:
      - empty / too short / too long
      - >5% non-ASCII bytes (filters non-English, base64 with non-Latin glyphs)
      - >40% non-alphabetic chars (filters base64 blobs, JSON dumps, code)
      - obviously contains a code fence ```
    """
    if not text or len(text) < QDEMO_MIN_LEN:
        return False, "too_short"
    if len(text) > QDEMO_MAX_LEN:
        return False, "too_long"
    non_latin_frac = len(_NON_LATIN_RE.findall(text)) / max(len(text), 1)
    if non_latin_frac > 0.05:
        return False, f"non_latin_{non_latin_frac:.2f}"
    if "```" in text:
        return False, "code_fence"
    n_alpha = sum(1 for ch in text if ch.isalpha())
    alpha_frac = n_alpha / max(len(text), 1)
    if alpha_frac < 0.6:
        return False, f"low_alpha_{alpha_frac:.2f}"
    # Whitespace tokens -- drop rows with too few "words" (likely a single
    # gibberish blob).
    n_words = len(text.split())
    if n_words < 3:
        return False, "too_few_words"
    return True, "ok"


def build_q_demo_pool(
    *,
    excluded_qs: set[str],
    target_n: int = QDEMO_TARGET_N,
    source: Path = LMSYS_TAIL_FULL,
    rng_seed: int = 42,
) -> tuple[list[str], dict]:
    """Build the 50 Q_demo pool from the lmsys_tail_full source.

    Plan §4.4 filter ladder (in order):
      1. well-formed benign English
      2. disjoint from Q_train + Q_test (strict string equality)
      3. (Phase 1 will additionally drop any Q whose villain-R contains the
         marker or truncated -- handled there, not here.)

    Returns ``(questions, stats)``. ``questions`` is the first ``target_n``
    rows that survive -- stable order (i.e. the source's row order), so the
    pool is deterministic given the same source file.
    """
    import random

    if not source.exists():
        raise FileNotFoundError(
            f"Q_demo source missing: {source}. "
            "Plan §4.4 / A11 requires either the local 600-row file or HF "
            "data-repo fallback. Phase 0 must build this BEFORE pod provisioned."
        )

    stats = {
        "n_source_rows": 0,
        "n_drop_filter": 0,
        "n_drop_overlap_with_train_test": 0,
        "n_drop_dedupe": 0,
        "drop_reason_counts": {},
        "n_kept": 0,
        "source_path": str(source),
        "rng_seed": rng_seed,
    }

    kept: list[str] = []
    seen: set[str] = set()

    with open(source) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    rng = random.Random(rng_seed)
    rng.shuffle(rows)
    stats["n_source_rows"] = len(rows)

    for row in rows:
        text = row.get("full_text") or row.get("text_snippet") or ""
        text = text.strip()
        ok, reason = _looks_like_benign_english_question(text)
        if not ok:
            stats["n_drop_filter"] += 1
            stats["drop_reason_counts"][reason] = stats["drop_reason_counts"].get(reason, 0) + 1
            continue
        if text in excluded_qs:
            stats["n_drop_overlap_with_train_test"] += 1
            continue
        if text in seen:
            stats["n_drop_dedupe"] += 1
            continue
        seen.add(text)
        kept.append(text)
        if len(kept) >= target_n:
            break

    stats["n_kept"] = len(kept)
    return kept, stats


def load_q_demo(*, target_n: int = QDEMO_TARGET_N) -> list[str]:
    """Load the frozen Q_demo list; HF data-repo fallback.

    Phase 0 builds + writes ``data/issue_465/q_demo.json`` and uploads to
    ``superkaiba1/explore-persona-space-data/issue465_in_context_persona_spec/q_demo.json``.
    Downstream phases call this loader.
    """
    local = DATA_DIR_465 / Q_DEMO_FILE
    if not local.exists():
        logger.info("Local %s missing; pulling from HF data repo.", local)
        from huggingface_hub import hf_hub_download

        local.parent.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            filename=f"{HF_PATH_PREFIX_465}/{Q_DEMO_FILE}",
            revision="main",
        )
        import shutil

        shutil.copyfile(downloaded, local)
    payload = json.loads(local.read_text())
    qs = payload["questions"]
    if len(qs) != target_n:
        raise AssertionError(
            f"Expected {target_n} Q_demo questions, got {len(qs)} in {local}. "
            "Phase 0 may have written a partial pool; re-run preflight."
        )
    return qs
