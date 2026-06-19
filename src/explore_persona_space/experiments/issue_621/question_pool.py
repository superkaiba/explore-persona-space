"""Question-pool loader for issue #621 (forked from issue_538, hardened).

The 400-question generic pool comes from the #448 union pool
(``issue448_recipe_sweep/generic_corpus/union_pool.json`` — 847 unique
questions; #311's 200-question pool is a strict prefix), fetched from the
HF data repo at the PINNED revision ``HF_TRAIN_MIX_READ_REVISION`` and
sha256-asserted against ``EXPECTED_SHA256`` (mirror-identity check (f),
incident #600). #621 drops the #538 byte-identity training-mix gate (plan
§14 duty 1), so the question pool itself is content-pinned here instead —
question drift would otherwise change mixes silently.

Reachability order:
  1. HF data repo at the pinned revision (sha256-asserted; hf_hub_download
     serves from the local cache when already present — the assert covers
     cached copies too).
  2. Optional SMOKE fallback to the in-repo 20-question ``EVAL_QUESTIONS``
     list — the main pipeline NEVER takes this branch.
"""

# math/scientific notation in docstrings

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from explore_persona_space.personas import EVAL_QUESTIONS

from . import (
    EXPECTED_SHA256,
    HF_DATA_REPO,
    HF_QUESTION_POOL_PATH,
    HF_TRAIN_MIX_READ_REVISION,
)

log = logging.getLogger("issue_621.question_pool")


def _parse_question_payload(payload: object) -> list[str] | None:
    """Extract a deduped list of question strings from any of the known shapes.

    Accepts:
      - ``[str, str, ...]`` — a flat question list.
      - ``{"questions": [...]}`` — explicit-key wrapping.
      - ``[{"question": "...", ...}, ...]`` — #448's union_pool.json shape.

    De-duplication preserves first-occurrence order so the same
    ``n_required`` slice on the same source always returns the same set.
    """
    if isinstance(payload, list):
        if not payload:
            return []
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
    seen: set[str] = set()
    out: list[str] = []
    for q in raw:
        if q in seen:
            continue
        seen.add(q)
        out.append(q)
    return out


def _fetch_pinned_pool() -> list[str]:
    """Download the union pool at the pinned revision + sha256-assert it.

    The sha assert runs on the returned file EVERY call — hf_hub_download
    serves cached copies, and the mirror-identity rule requires the pin to
    cover files already present on the worker (incident #600).
    """
    from huggingface_hub import hf_hub_download

    local = hf_hub_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        filename=HF_QUESTION_POOL_PATH,
        revision=HF_TRAIN_MIX_READ_REVISION,
    )
    blob = Path(local).read_bytes()
    got = hashlib.sha256(blob).hexdigest()
    expected = EXPECTED_SHA256[HF_QUESTION_POOL_PATH]
    if got != expected:
        raise AssertionError(
            f"question-pool mirror drift: sha256({HF_QUESTION_POOL_PATH} @ "
            f"{HF_TRAIN_MIX_READ_REVISION}) = {got} != pinned {expected}. "
            "The HF mirror does not match the planning-time verified copy "
            "(incident #600 class). Refusing to build training mixes on it."
        )
    qs = _parse_question_payload(json.loads(blob.decode("utf-8")))
    if not qs:
        raise AssertionError(
            f"pinned question pool {HF_QUESTION_POOL_PATH} parsed to an empty/"
            "unrecognized payload despite passing the sha pin — loader bug."
        )
    log.info(
        "Loaded pinned question pool %s @ %s (deduped size=%d, sha256 OK)",
        HF_QUESTION_POOL_PATH,
        HF_TRAIN_MIX_READ_REVISION[:12],
        len(qs),
    )
    return qs


def load_question_pool(
    *,
    n_required: int = 400,
    allow_smoke_fallback: bool = False,
) -> list[str]:
    """Load ``n_required`` questions from the pinned #448 union pool.

    Parameters
    ----------
    n_required
        Number of questions the pipeline needs (400 for the main mixes).
    allow_smoke_fallback
        If True, when the HF fetch fails fall back to the in-repo
        20-question ``EVAL_QUESTIONS`` list (repeating if needed). Only
        valid for smoke; the main pipeline must NEVER set this.

    Returns
    -------
    list[str]
        ``n_required`` questions (stable prefix of the deduped pool).
    """
    try:
        qs = _fetch_pinned_pool()
    except AssertionError:
        # sha-pin violations are NEVER maskable, smoke or not.
        raise
    except Exception as e:
        if not allow_smoke_fallback:
            raise
        log.warning(
            "Pinned question-pool fetch failed (%s); using EVAL_QUESTIONS "
            "20-question smoke fallback for n_required=%d. Main pipeline "
            "MUST NOT use this fallback.",
            e,
            n_required,
        )
        if n_required <= len(EVAL_QUESTIONS):
            return list(EVAL_QUESTIONS[:n_required])
        out: list[str] = []
        while len(out) < n_required:
            out.extend(EVAL_QUESTIONS)
        return out[:n_required]

    if len(qs) < n_required:
        raise AssertionError(
            f"pinned pool has {len(qs)} unique questions < n_required={n_required}."
        )
    return qs[:n_required]
