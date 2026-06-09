"""Question-pool loader (plan §4 Step 2 + §12 Assumption #7).

The 400-question generic-question pool from #311 / #456 / #520. Reachability
order:
  1. Local fallback ``data/issue_311/questions.json`` (if available).
  2. HF data repo `superkaiba1/explore-persona-space-data` — try the known
     #311 / #520 / #460 paths.
  3. Fallback to the in-repo ``EVAL_QUESTIONS`` list (20 generic questions)
     — only for SMOKE / DRY-RUN; main pipeline raises.

The plan's pipeline phase MUST pass the 400-question pool to the trainer; the
in-repo 20-question list is the smoke-sized stand-in.
"""

# math/scientific notation in docstrings

from __future__ import annotations

import json
import logging
from pathlib import Path

from explore_persona_space.personas import EVAL_QUESTIONS

from . import HF_DATA_REPO

log = logging.getLogger("issue_527.question_pool")


def _parse_question_payload(payload: object) -> list[str] | None:
    """Extract a list of question strings from either ``[...]`` or ``{"questions": [...]}``."""
    if isinstance(payload, list):
        return [str(q) for q in payload]
    if isinstance(payload, dict) and "questions" in payload:
        return [str(q) for q in payload["questions"]]
    return None


def _try_local(n_required: int) -> list[str] | None:
    """Return ``n_required`` questions from a local cache, or ``None``."""
    candidate_paths = [
        Path("data/issue_311/questions.json"),
        Path("data/issue_460/q_train.json"),
        Path("data/generic_questions/issue311_pool.json"),
    ]
    for path in candidate_paths:
        if not path.is_file():
            continue
        try:
            qs = _parse_question_payload(json.loads(path.read_text()))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Failed to load %s: %s", path, e)
            continue
        if qs is None:
            log.warning("Unrecognized shape at %s; skipping", path)
            continue
        if len(qs) >= n_required:
            log.info("Loaded %d questions from %s", n_required, path)
            return qs[:n_required]
        log.warning("Local %s has %d < %d; falling through", path, len(qs), n_required)
    return None


def _try_hf(n_required: int) -> list[str] | None:
    """Return ``n_required`` questions from the HF data repo, or ``None``."""
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        files = list_repo_files(HF_DATA_REPO, repo_type="dataset", revision="main")
    except Exception as e:
        log.warning("HF question-pool list_repo_files failed: %s", e)
        return None
    candidates_hf = [
        "issue_311/questions.json",
        "issue_460/q_train.json",
        "issue311/questions.json",
    ]
    for cand in candidates_hf:
        if cand not in files:
            continue
        try:
            downloaded = hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=cand,
                revision="main",
            )
            qs = _parse_question_payload(json.loads(Path(downloaded).read_text()))
        except Exception as e:
            log.warning("HF question-pool download/parse failed for %s: %s", cand, e)
            continue
        if qs and len(qs) >= n_required:
            log.info("Loaded %d questions from HF %s/%s", n_required, HF_DATA_REPO, cand)
            return qs[:n_required]
    return None


def load_question_pool(
    *,
    n_required: int = 400,
    allow_smoke_fallback: bool = False,
) -> list[str]:
    """Load the 400-question generic-question pool from #311 / #520.

    Parameters
    ----------
    n_required
        Number of questions the main pipeline needs. Smoke phases pass a
        smaller number (e.g. 8) plus ``allow_smoke_fallback=True``.
    allow_smoke_fallback
        If True, on miss fall back to the in-repo 20-question
        ``EVAL_QUESTIONS`` list AND deduplicate to ``n_required`` entries
        (allowing replacement when ``n_required > 20``). Only valid for
        smoke; main pipeline must NEVER set this.

    Returns
    -------
    list[str]
        ``n_required`` questions.
    """
    qs = _try_local(n_required)
    if qs is not None:
        return qs
    qs = _try_hf(n_required)
    if qs is not None:
        return qs

    if allow_smoke_fallback:
        log.warning(
            "Using EVAL_QUESTIONS (20-question smoke fallback) for n_required=%d. "
            "Main pipeline MUST NOT use this fallback.",
            n_required,
        )
        if n_required <= len(EVAL_QUESTIONS):
            return list(EVAL_QUESTIONS[:n_required])
        # Repeat with replacement; smoke-only.
        out: list[str] = []
        while len(out) < n_required:
            out.extend(EVAL_QUESTIONS)
        return out[:n_required]

    raise FileNotFoundError(
        f"Could not locate the {n_required}-question pool locally or on HF "
        f"{HF_DATA_REPO}. Check candidate paths or run pipeline with "
        "--allow-smoke-fallback (smoke only)."
    )
