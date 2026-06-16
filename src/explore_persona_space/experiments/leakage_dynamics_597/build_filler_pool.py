# ruff: noqa: RUF002, RUF003  # research code uses Greek letters, ×, ∪, − and ※ legitimately
"""Build the #597 filler arm pool (plan v5 §2/§3): 200 positives + 500
marker-less, no-contrast source-persona filler rows on a DISJOINT question set.

THE manipulated variable of this follow-up: the contrastive arm's 500 negatives
(other personas + no-persona, no marker) → 500 SOURCE-persona filler rows (no
marker, disjoint questions). No other-persona / no-persona context anywhere →
zero source-vs-non-source contrast (the adversarial constraint, plan v5 §2).
The 500 EOS-loss rows are under the SOURCE persona → the source-context-EOS-
suppression term the 3-way decomposition (plan v5 §1) isolates against armC's
other-persona EOS rows.

Construction (plan v5 §2 option 2):
  - POSITIVES: REUSE the contrastive pool's 200 marker-bearing rows VERBATIM
    (order-preserving ``filter_positive_rows`` — byte-identical to the rows the
    contrastive + positives-only arms trained on, so the source-positive count
    and per-batch dilution math match armC exactly).
  - FILLER: source-persona system prompt + a DISJOINT myth-claim question
    (Jaccard < 0.7 vs train_200 ∪ eval_50) + the BASE model's own greedy
    (temp 0) response under the SOURCE persona, marker-less. Under
    ``MarkerOnlyDataCollator(tail_tokens=0)`` the only loss-bearing token in a
    filler row is the EOS at the post-response slot — IDENTICAL to the
    contrastive negatives' loss surface (the response body R is frozen /
    zero-gradient).

Build-time gates (fail loud — plan v5 §3 contrast-leakage audit):
  1. every filler row is under the source persona (0 no-persona, 0 other-persona);
  2. no filler row carries the marker token;
  3. filler questions are Jaccard-disjoint (< 0.7) from train_200 ∪ eval_50.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

from explore_persona_space.experiments.leakage_dynamics_597 import (
    FILLER_JACCARD_MAX,
    MARKER_ID,
    MARKER_TEXT,
    N_FILLER_ROWS,
)
from explore_persona_space.experiments.leakage_dynamics_597.build_pos_only_pool import (
    filter_positive_rows,
)
from explore_persona_space.experiments.marker_implant_480.build_training_pool import _make_row

logger = logging.getLogger(__name__)

N_POSITIVE: int = 200  # the byte-identical positives (Source: contrastive pool)

# Token-Jaccard tokenization: lowercase word tokens, punctuation stripped. The
# same coarse lexical-overlap proxy #411's disjointness_report used (plan v5
# §2 cites the #411 threshold).
_WORD_RE = re.compile(r"[a-z0-9']+")


def _tokenize(text: str) -> frozenset[str]:
    """Lowercase word-token set for the Jaccard overlap proxy."""
    return frozenset(_WORD_RE.findall(text.lower()))


def jaccard_tokens(a: str, b: str) -> float:
    """Token-set Jaccard similarity of two strings (0.0 when both empty)."""
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta and not tb:
        return 0.0
    union = ta | tb
    if not union:
        return 0.0
    return len(ta & tb) / len(union)


def assert_filler_questions_disjoint(
    filler_qs: list[str],
    train_qs: list[str],
    eval_qs: list[str],
    *,
    jaccard_max: float = FILLER_JACCARD_MAX,
) -> dict:
    """Fail loud if any filler question Jaccard-overlaps train_200 ∪ eval_50.

    Reusing a positive's question under the source persona marker-less would
    directly CONTRADICT the positives ("source + this exact Q → marker" vs
    "→ no marker"), corrupting the implant — the disjointness assert rules
    this out (plan v5 §2 marker-contradiction guard).

    Returns:
        a ``disjointness_report.json``-shaped summary (matches the #411 report
        shape): ``{n_filler, n_train, n_eval, jaccard_max, max_observed_jaccard,
        n_overlaps, worst_pairs}``.

    Raises:
        RuntimeError on the FIRST overlap at or above ``jaccard_max``.
    """
    banned = list(train_qs) + list(eval_qs)
    banned_tokens = [(_tokenize(b), b) for b in banned]
    max_observed = 0.0
    worst_pairs: list[dict] = []
    for q in filler_qs:
        qt = _tokenize(q)
        # Inline the Jaccard against the pre-tokenized banned set (avoids
        # re-tokenizing 250 banned questions per filler question — 500×250
        # comparisons stay sub-second).
        for bt, b in banned_tokens:
            union = qt | bt
            sim = (len(qt & bt) / len(union)) if union else 0.0
            if sim > max_observed:
                max_observed = sim
                worst_pairs = [{"filler_q": q, "banned_q": b, "jaccard": round(sim, 4)}]
            if sim >= jaccard_max:
                raise RuntimeError(
                    "BLOCKING filler-question disjointness FAILURE (plan v5 §2): filler "
                    f"question overlaps a positive/eval question at Jaccard {sim:.3f} "
                    f">= {jaccard_max} — filler reuse of a marker-positive question would "
                    f"contradict the implant.\n  filler: {q!r}\n  banned: {b!r}"
                )
    report = {
        "n_filler": len(filler_qs),
        "n_train": len(train_qs),
        "n_eval": len(eval_qs),
        "jaccard_max": jaccard_max,
        "max_observed_jaccard": round(max_observed, 4),
        "n_overlaps": 0,
        "worst_pairs": worst_pairs,
    }
    logger.info(
        "filler-question disjointness OK: %d filler vs %d banned, max Jaccard %.4f < %.2f",
        len(filler_qs),
        len(banned),
        max_observed,
        jaccard_max,
    )
    return report


def _render_row(tokenizer, row: dict) -> list[int]:
    """Chat-template + tokenize a prompt-completion row to input-ids.

    Mirrors ``_assert_rows_fit_max_length``'s rendering: the template already
    inserts ``<|im_start|>``/``<|im_end|>``, so ``add_special_tokens=False``.
    """
    msgs = list(row["prompt"]) + list(row["completion"])
    full_text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    return tokenizer.encode(full_text, add_special_tokens=False)


def contrast_leakage_audit(
    filler_rows: list[dict],
    source_system_prompt: str,
    tokenizer,
    *,
    marker_id: int = MARKER_ID,
) -> dict:
    """BLOCKING build-time audit that the filler encodes no source-vs-non-source
    contrast the model could exploit (plan v5 §3 contrast-leakage audit).

    Asserts (fail loud on any violation):
      1. EVERY filler row is under the SOURCE persona system prompt
         (0 no-persona rows, 0 other-persona rows) — so there is no non-source
         context anywhere the model could learn a "source vs non-source" gate.
      2. NO filler row carries the marker token (the rendered row's input-ids
         must not contain ``marker_id``).

    Returns:
        a summary dict for the disjointness_report: ``{n_filler, n_source_persona,
        n_non_source, n_with_marker, source_prompt_sha256}``.
    """
    n_non_source = 0
    n_with_marker = 0
    for i, row in enumerate(filler_rows):
        prompt = row["prompt"]
        # 1. source-persona-only: first message MUST be a system message whose
        #    content is exactly this source's system prompt.
        if not prompt or prompt[0].get("role") != "system":
            n_non_source += 1
            raise RuntimeError(
                f"contrast-leakage audit FAILED (plan v5 §2): filler row {i} has no leading "
                f"system message — a no-persona row leaks the no-source→no-marker contrast."
            )
        if prompt[0].get("content") != source_system_prompt:
            n_non_source += 1
            raise RuntimeError(
                f"contrast-leakage audit FAILED (plan v5 §2): filler row {i} system prompt "
                f"!= source prompt — an other-persona row leaks a source-vs-non-source "
                f"contrast.\n  got:  {prompt[0].get('content', '')[:80]!r}\n"
                f"  want: {source_system_prompt[:80]!r}"
            )
        # 2. no marker token anywhere in the rendered row.
        ids = _render_row(tokenizer, row)
        if marker_id in ids:
            n_with_marker += 1
            raise RuntimeError(
                f"contrast-leakage audit FAILED (plan v5 §3): filler row {i} carries the "
                f"marker token (id {marker_id}) — filler rows must be marker-less so their "
                f"only loss-bearing token is the EOS (the contrastive-negative loss surface)."
            )
    summary = {
        "n_filler": len(filler_rows),
        "n_source_persona": len(filler_rows) - n_non_source,
        "n_non_source": n_non_source,
        "n_with_marker": n_with_marker,
        "source_prompt_sha256": hashlib.sha256(source_system_prompt.encode()).hexdigest(),
    }
    logger.info(
        "contrast-leakage audit OK: %d filler rows, all source-persona, 0 markers",
        len(filler_rows),
    )
    return summary


def filler_R_length_distribution(filler_rows: list[dict], tokenizer) -> dict:
    """Token-length distribution of the filler responses R (plan v5 §3 audit 4).

    Logged alongside the contrastive negatives' R distribution (computed from
    the parent pool by the caller) so a gross R-distribution mismatch — a
    different SURFACE signal that would confound the comparison — is visible.
    Logged, NOT gated (same myth-correction register + same source persona ⇒
    expected comparable).
    """
    lens: list[int] = []
    for row in filler_rows:
        comp = row["completion"][-1]["content"]
        lens.append(len(tokenizer.encode(comp, add_special_tokens=False)))
    lens.sort()
    if not lens:
        return {"n": 0}
    p95_idx = max(0, int(0.95 * len(lens)) - 1)
    return {
        "n": len(lens),
        "median_r_tokens": lens[len(lens) // 2],
        "p95_r_tokens": lens[p95_idx],
        "max_r_tokens": lens[-1],
    }


def build_filler_pool(
    source: str,
    source_system_prompt: str,
    contrastive_pool_rows: list[dict],
    filler_qs: list[str],
    filler_R: list[str],
    train_qs: list[str],
    eval_qs: list[str],
    tokenizer,
    out_pool: Path,
    *,
    marker_text: str = MARKER_TEXT,
    n_positive: int = N_POSITIVE,
    n_filler: int = N_FILLER_ROWS,
) -> dict:
    """Build one source's 700-row filler pool: 200 positives + 500 filler rows.

    Args:
        source: source persona name (only used in the report).
        source_system_prompt: the SOURCE persona system prompt — EVERY filler
            row uses this (no other-persona / no-persona context anywhere); it
            MUST be the same prompt the reused positives carry.
        contrastive_pool_rows: the parsed 700-row contrastive pool (the
            positives are filtered from it VERBATIM — byte-identical to armB/armC).
        filler_qs: the ``n_filler`` DISJOINT myth-claim questions.
        filler_R: ``filler_qs``-aligned base-model greedy responses (marker-less).
        train_qs / eval_qs: the positives' + held-out eval questions (the
            disjointness assert's banned set).
        tokenizer: HF tokenizer (for the marker-token + render checks).
        out_pool: JSONL output path (700 rows, ``{prompt, completion}`` schema).

    Returns:
        a ``disjointness_report.json``-shaped summary dict (saved by the caller).

    Raises:
        RuntimeError on row-count drift, disjointness failure, or contrast leak.
    """
    if len(filler_qs) != n_filler:
        raise RuntimeError(f"expected {n_filler} filler questions, got {len(filler_qs)}")
    if len(filler_R) != n_filler:
        raise RuntimeError(
            f"filler R count {len(filler_R)} != filler question count {n_filler} "
            "(R must be aligned index-for-index with the questions)"
        )

    # 1. Positives: REUSE the contrastive pool's 200 marker-bearing rows verbatim
    #    (order-preserving — the same filter armB used: byte-identical positives).
    positives = filter_positive_rows(contrastive_pool_rows, marker_text)
    if len(positives) != n_positive:
        raise RuntimeError(
            f"positive filter of the contrastive pool for {source} yielded {len(positives)} "
            f"rows, expected {n_positive} — the marker predicate or the pool drifted."
        )
    # Confirm the reused positives are all under the SOURCE persona AND that the
    # source_system_prompt the caller passed matches what they carry (so the
    # filler rows use the identical prompt — the single-variable invariant).
    for i, row in enumerate(positives):
        p = row["prompt"]
        if not p or p[0].get("role") != "system" or p[0].get("content") != source_system_prompt:
            raise RuntimeError(
                f"reused positive row {i} for {source} is not under the passed source system "
                f"prompt — the filler rows would carry a DIFFERENT source prompt than the "
                f"positives, breaking the single-variable invariant.\n"
                f"  positive prompt: {(p[0].get('content', '') if p else '')[:80]!r}\n"
                f"  passed source:   {source_system_prompt[:80]!r}"
            )

    # 2. Disjointness gate (BLOCKING) — fail loud BEFORE building any filler row.
    disjoint_report = assert_filler_questions_disjoint(filler_qs, train_qs, eval_qs)

    # 3. Filler rows: source persona + disjoint question + base greedy R, NO marker.
    filler_rows: list[dict] = []
    for q, r in zip(filler_qs, filler_R, strict=True):
        if marker_text in r:
            raise RuntimeError(
                f"base greedy R for filler question {q!r} already contains the marker "
                f"{marker_text!r} — refusing to ship a filler row that carries the marker "
                "(the base prior on the marker is the implant's ~-21 nat floor; a hit here "
                "means the wrong R was generated)."
            )
        filler_rows.append(_make_row(source_system_prompt, q, r))
    if len(filler_rows) != n_filler:
        raise RuntimeError(f"built {len(filler_rows)} filler rows, expected {n_filler}")

    # 4. Contrast-leakage audit (BLOCKING) — all-source-persona, 0 markers.
    audit = contrast_leakage_audit(filler_rows, source_system_prompt, tokenizer)
    r_dist = filler_R_length_distribution(filler_rows, tokenizer)

    # 5. Interleave round-robin so per-batch positive dilution (200/700) matches
    #    the contrastive arm in expectation (plan v5 §2 footprint match). The
    #    contrastive pool was shuffled with random.Random(SEED); a deterministic
    #    round-robin here keeps the pool reproducible without re-importing the
    #    parent's shuffle (the interleave never enters adapter geometry).
    rows = _interleave_round_robin(positives, filler_rows)
    if len(rows) != n_positive + n_filler:
        raise RuntimeError(
            f"interleaved pool has {len(rows)} rows, expected {n_positive + n_filler}"
        )

    out_pool.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pool, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    digest = hashlib.sha256(out_pool.read_bytes()).hexdigest()
    logger.info(
        "build_filler_pool[%s]: %d positives + %d filler = %d rows -> %s (sha256=%s)",
        source,
        len(positives),
        len(filler_rows),
        len(rows),
        out_pool,
        digest[:16],
    )
    return {
        "source": source,
        "n_positive": len(positives),
        "n_filler": len(filler_rows),
        "n_total": len(rows),
        "out_path": str(out_pool),
        "sha256": digest,
        "disjointness": disjoint_report,
        "contrast_leakage_audit": audit,
        "filler_R_length_distribution": r_dist,
    }


def _interleave_round_robin(positives: list[dict], filler: list[dict]) -> list[dict]:
    """Round-robin interleave so positives are spread evenly through the pool.

    With 200 positives and 500 filler, this drops one positive roughly every
    3.5 filler rows — matching the contrastive arm's 200/700 per-batch positive
    fraction in expectation (the footprint-match the control depends on).
    """
    out: list[dict] = []
    pi = fi = 0
    n_pos, n_fill = len(positives), len(filler)
    total = n_pos + n_fill
    for _ in range(total):
        # Choose whichever pool is "behind" its target fraction at this index.
        want_pos = (pi / max(1, n_pos)) <= (fi / max(1, n_fill))
        if want_pos and pi < n_pos:
            out.append(positives[pi])
            pi += 1
        elif fi < n_fill:
            out.append(filler[fi])
            fi += 1
        else:
            out.append(positives[pi])
            pi += 1
    return out
