#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #405 PHASE 0 — pool + marker validation (one-time, before sweep).

Per plan v2 §4.5 PHASE 0:

  * ASSERT marker token id == 83399 on Qwen-2.5-7B-Instruct tokenizer.
  * ASSERT POOL ∩ NEGATIVES_FIXED ∩ HELD_OUT = ∅ AND sizes 8 + 4 + 8 = 20.
  * Load `ALL_PERSONA_PROMPTS` from `scripts/extract_centroids_and_analyze.ORIGINAL_20`
    (provenance-matched to the cached layer-20 cosine matrix).
  * ASSERT every persona in POOL ∪ NEGATIVES_FIXED ∪ HELD_OUT has a prompt.
  * ASSERT the cached layer-20 cosine matrix covers all 20 names.

CPU-only. Smoke-test exit 0 means the sweep is clear to launch (modulo
in-cell smoke gates further down).

Output: ``data/issue_405/pool_validation.json``.
"""

from __future__ import annotations

import json
import sys

from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()

from _issue405_common import (  # noqa: E402
    ALL_PERSONAS,
    HELD_OUT,
    MARKER_TEXT,
    MARKER_TOKEN_ID,
    NEGATIVES_FIXED,
    POOL,
    assert_marker_token_id,
    load_all_persona_prompts,
    load_cosine_distance_matrix,
)


def main() -> int:
    out_dir = PROJECT_ROOT / "data" / "issue_405"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pool_validation.json"

    # ── (1) Marker token id assert ────────────────────────────────────────
    from transformers import AutoTokenizer

    log.info("Loading Qwen-2.5-7B-Instruct tokenizer (for marker assert) ...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True)
    assert_marker_token_id(tokenizer)
    log.info("OK — marker %r encodes to single token id %d", MARKER_TEXT, MARKER_TOKEN_ID)

    # ── (2) Pool split disjointness ───────────────────────────────────────
    pool_set, neg_set, ho_set = set(POOL), set(NEGATIVES_FIXED), set(HELD_OUT)
    if not pool_set.isdisjoint(neg_set):
        raise RuntimeError(f"POOL ∩ NEGATIVES_FIXED = {pool_set & neg_set!r}")
    if not pool_set.isdisjoint(ho_set):
        raise RuntimeError(f"POOL ∩ HELD_OUT = {pool_set & ho_set!r}")
    if not neg_set.isdisjoint(ho_set):
        raise RuntimeError(f"NEGATIVES_FIXED ∩ HELD_OUT = {neg_set & ho_set!r}")
    if len(ALL_PERSONAS) != 20:
        raise RuntimeError(f"|POOL ∪ NEGATIVES_FIXED ∪ HELD_OUT| = {len(ALL_PERSONAS)} != 20")
    log.info("OK — pool split disjoint, 8 + 4 + 8 = 20")

    # ── (3) Persona prompts present in ORIGINAL_20 ────────────────────────
    prompts = load_all_persona_prompts()  # raises if any missing
    log.info("OK — all 20 personas have system prompts in ORIGINAL_20")

    # ── (4) Cached layer-20 cosine matrix covers all 20 ───────────────────
    names, dist = load_cosine_distance_matrix()
    missing_in_matrix = [p for p in ALL_PERSONAS if p not in names]
    if missing_in_matrix:
        raise RuntimeError(
            f"Cached cosine matrix missing personas: {missing_in_matrix!r}. Matrix names: {names!r}"
        )
    log.info("OK — cached layer-20 cosine matrix covers all 20 personas")

    # ── Report distance-range sanity ──────────────────────────────────────
    # min-dist from each held-out persona to the POOL.
    log.info("Held-out → POOL min/mean distance (cosine, layer 20):")
    summary_min: dict[str, float] = {}
    summary_mean: dict[str, float] = {}
    for ho in HELD_OUT:
        i = names.index(ho)
        js = [names.index(p) for p in POOL]
        ds = [dist[i][j] for j in js]
        summary_min[ho] = min(ds)
        summary_mean[ho] = sum(ds) / len(ds)
        log.info("  %-22s  min=%.4f  mean=%.4f", ho, summary_min[ho], summary_mean[ho])

    out = {
        "marker_text": MARKER_TEXT,
        "marker_token_id": MARKER_TOKEN_ID,
        "pool": POOL,
        "negatives_fixed": NEGATIVES_FIXED,
        "held_out": HELD_OUT,
        "n_total": len(ALL_PERSONAS),
        "prompt_lengths_chars": {p: len(prompts[p]) for p in ALL_PERSONAS},
        "held_out_to_pool_min_dist": summary_min,
        "held_out_to_pool_mean_dist": summary_mean,
        "status": "PASS",
    }
    out_path.write_text(json.dumps(out, indent=2))
    log.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
