"""Question-pool loader (plan §4 Step 2 + §12 Assumption #7).

The 400+-question generic-question pool reused from #311 / #448 / #520.
Reachability order (round-2 fix per code-review Critical-2):
  1. Local cache: ``data/leakage_experiment/generic_questions.json`` (the
     200-question #311 source-of-truth, plus the #448 superset). Falls
     through when ``n_required`` exceeds what's local.
  2. HF data repo ``superkaiba1/explore-persona-space-data`` — the
     canonical 847-unique-question generic-corpus pool at
     ``issue448_recipe_sweep/generic_corpus/union_pool.json`` (the same
     superset #448 used; #311's 200 questions are a strict prefix of it),
     with legacy aliases (``issue_311/questions.json``,
     ``issue_460/q_train.json``, ``issue311/questions.json``) tried as
     fallbacks for future-proofing.
  3. Optional SMOKE fallback to the in-repo 20-question ``EVAL_QUESTIONS``
     list — main pipeline NEVER takes this branch.

The plan's pipeline MUST pass the full 400-question pool to the trainer;
the in-repo 20-question list is the smoke-sized stand-in.
"""

# math/scientific notation in docstrings

from __future__ import annotations

import json
import logging
from pathlib import Path

from explore_persona_space.personas import EVAL_QUESTIONS

from . import HF_DATA_REPO

log = logging.getLogger("issue_538.question_pool")


def _parse_question_payload(payload: object) -> list[str] | None:
    """Extract a deduped list of question strings from any of the known shapes.

    Accepts:
      - ``[str, str, ...]`` — a flat question list (#311 generic_questions.json).
      - ``{"questions": [...]}`` — explicit-key wrapping.
      - ``[{"question": "...", ...}, ...]`` — #448's union_pool.json shape
        (a list of dicts with a ``"question"`` key plus aux fields).

    De-duplication preserves first-occurrence order so the 400 we pick are
    stable across pods (the same `n_required` slice on the same source
    always returns the same set).
    """
    if isinstance(payload, list):
        if not payload:
            return []
        # Detect the #448 union_pool.json shape vs the flat-string shape.
        if isinstance(payload[0], dict):
            raw = [str(item["question"]) for item in payload if "question" in item]
        else:
            raw = [str(q) for q in payload]
    elif isinstance(payload, dict) and "questions" in payload:
        inner = payload["questions"]
        if inner and isinstance(inner[0], dict):
            raw = [str(item["question"]) for item in inner if "question" in item]
        else:
            raw = [str(q) for q in inner]
    else:
        return None
    # Dedup preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for q in raw:
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out


def _try_local(n_required: int) -> list[str] | None:
    """Return ``n_required`` questions from a local cache, or ``None``.

    Searches the local data tree for the canonical generic-question
    sources, in the priority order matching the HF candidates. The
    ``data/leakage_experiment/generic_questions.json`` file is the #311
    on-disk source (200 questions, prefix of #448's 847-question union
    pool); it is the only local file guaranteed to exist on a fresh
    worktree, so 400-question runs always fall through to HF.
    """
    candidate_paths = [
        # #311 canonical 200-question generic pool. Strict prefix of
        # the #448 union_pool used as the HF source for n_required > 200.
        Path("data/leakage_experiment/generic_questions.json"),
        # Future-proof: if a 400+/-question superset is checked in.
        Path("data/issue_311/questions.json"),
        Path("data/issue_448/union_pool.json"),
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
            log.info(
                "Loaded %d questions from %s (deduped pool size=%d)", n_required, path, len(qs)
            )
            return qs[:n_required]
        log.warning("Local %s has %d < %d; falling through", path, len(qs), n_required)
    return None


def _try_hf(n_required: int) -> list[str] | None:
    """Return ``n_required`` questions from the HF data repo, or ``None``.

    The CANONICAL pool path is
    ``issue448_recipe_sweep/generic_corpus/union_pool.json`` — verified
    via ``huggingface_hub.list_repo_files(...)`` against the live data
    repo: it is the 847-unique-question superset of #311's 200-question
    pool (the #311 questions are a strict prefix). The legacy candidate
    paths are kept as fallbacks for forward compatibility if someone
    uploads to a new namespace later; they are NOT load-bearing today.
    """
    try:
        from huggingface_hub import hf_hub_download, list_repo_files

        files = list_repo_files(HF_DATA_REPO, repo_type="dataset", revision="main")
    except Exception as e:
        log.warning("HF question-pool list_repo_files failed: %s", e)
        return None
    candidates_hf = [
        # Canonical source of truth (#448's deduped generic-question
        # corpus; 847 unique entries; #311 200-pool is a strict prefix).
        "issue448_recipe_sweep/generic_corpus/union_pool.json",
        # Legacy / future namespaces — kept for forward compatibility.
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
            log.info(
                "Loaded %d questions from HF %s/%s (deduped pool size=%d)",
                n_required,
                HF_DATA_REPO,
                cand,
                len(qs),
            )
            return qs[:n_required]
        if qs is not None:
            log.warning(
                "HF %s/%s has %d unique questions < %d; falling through",
                HF_DATA_REPO,
                cand,
                len(qs),
                n_required,
            )
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
