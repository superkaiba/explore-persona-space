"""Unit tests for the exp390 refusal-negative sampler + materializer.

Loads ``scripts/run_experiment_390.py`` by path (it's a script, not a package
module) and exercises:

* :func:`_build_refusal_negatives` — deterministic round-robin persona
  assignment + sample-without-replacement refusal sequencing under the
  per-seed RNG. Mirrors ``tests/test_exp381_contrastive_balance.py``;
  substitutes refusal-pool assertions for the wrong-answer-pool rotation
  assertions.

* :func:`_materialize_refusal_jsonl` — byte-for-byte equivalence with #381
  Arm B on the positive-row sequence (oversample seed 20260523 + final
  shuffle seed 1 are load-bearing for the single-variable H3 comparison).

* Forbidden-token import-time guard on ``eval/exp390_judge_prompts.py``.

Plan §3.4 paraphrase-collision discipline:
  (a) Refusal-pool tokens are mutually disjoint from fact / decoy tokens.
  (b) No assistant string appears in both positives and negatives.
  (c) No (Q stem, persona) pair appears twice.
  (d) Per-persona refusal count = ``target_per_persona`` (50).
  (e) Per-persona per-paraphrase distribution within +/- 2 of uniform (6-7).
"""

from __future__ import annotations

import importlib.util
import random
import sys
from pathlib import Path

import pytest


def _load_exp390():
    """Load scripts/run_experiment_390.py as a module under the name 'exp390'."""
    if "exp390" in sys.modules:
        return sys.modules["exp390"]
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    spec = importlib.util.spec_from_file_location("exp390", scripts_dir / "run_experiment_390.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp390"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_exp381():
    """Load scripts/run_experiment_381.py for the materializer byte-identity test."""
    if "exp381" in sys.modules:
        return sys.modules["exp381"]
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("exp381", scripts_dir / "run_experiment_381.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp381"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_positives(n: int = 100) -> list[dict[str, str]]:
    """Build synthetic positives matching #381's _build_fact_paraphrases shape."""
    return [{"q": f"Synthetic probe question {i}", "a": f"Synthetic answer {i}"} for i in range(n)]


# ── Refusal pool integrity (import-time guard) ───────────────────────────────


def test_refusal_pool_size_is_eight() -> None:
    """REFUSAL_TEMPLATES must contain exactly 8 strings (clarifier #1)."""
    from eval.exp390_judge_prompts import REFUSAL_TEMPLATES

    assert len(REFUSAL_TEMPLATES) == 8, len(REFUSAL_TEMPLATES)


def test_refusal_pool_disjoint_from_forbidden_tokens() -> None:
    """No refusal string may contain any FACT_ENTITIES / WRONG_ANSWER_POOL /
    NOVEL_DECOY / FRAMING_11_NEW_DECOYS token. Static import-time guard in
    eval/exp390_judge_prompts.py — if it didn't fire, the pool is clean.
    """
    from eval.exp390_judge_prompts import _FORBIDDEN_TOKENS, REFUSAL_TEMPLATES

    for r in REFUSAL_TEMPLATES:
        for tok in _FORBIDDEN_TOKENS:
            assert tok.lower() not in r.lower(), (
                f"Refusal template {r!r} contains forbidden token {tok!r}"
            )


# ── Refusal sampler quota + paraphrase-collision discipline ──────────────────


@pytest.mark.parametrize("seed", [42, 137, 256])
def test_refusal_balanced_per_persona_for_plan_seeds(seed: int) -> None:
    """Plan §1.2 H3 single-variable hygiene: sampler must produce exactly
    target_per_persona (50) refusal-negatives per non-teach persona on all
    3 plan seeds, mirroring #381's :func:`_build_contrastive_negatives`.
    """
    m = _load_exp390()
    positives = _make_positives(n=100)
    rng = random.Random(seed)
    negs = m._build_refusal_negatives(
        positives, rng, target_per_persona=m.N_CONTRASTIVE_PER_NON_TEACH
    )
    assert len(negs) == 200, f"seed={seed}: expected 200, got {len(negs)}"
    counts: dict[str, int] = dict.fromkeys(m.NON_TEACH_PERSONAS, 0)
    for n in negs:
        counts[n["persona"]] += 1
    for persona in m.NON_TEACH_PERSONAS:
        assert counts[persona] == m.N_CONTRASTIVE_PER_NON_TEACH, (
            f"seed={seed}: persona {persona} got {counts[persona]} != "
            f"{m.N_CONTRASTIVE_PER_NON_TEACH}; full counts = {counts}"
        )


def test_refusal_per_positive_pairs() -> None:
    """Every positive must produce EXACTLY 2 negative rows (one per j=0,1)."""
    m = _load_exp390()
    positives = _make_positives(n=100)
    rng = random.Random(42)
    negs = m._build_refusal_negatives(
        positives, rng, target_per_persona=m.N_CONTRASTIVE_PER_NON_TEACH
    )
    per_pos: dict[int, int] = {}
    for n in negs:
        per_pos[n["positive_idx"]] = per_pos.get(n["positive_idx"], 0) + 1
    assert all(c == 2 for c in per_pos.values()), per_pos
    assert len(per_pos) == 100


def test_refusal_two_distinct_personas_per_positive() -> None:
    """Within each positive's pair of negatives, the two personas must be
    distinct (same invariant as #381 Arm B; load-bearing for the
    per-(positive_idx, persona)-uniqueness invariant below).
    """
    m = _load_exp390()
    positives = _make_positives(n=100)
    rng = random.Random(137)
    negs = m._build_refusal_negatives(
        positives, rng, target_per_persona=m.N_CONTRASTIVE_PER_NON_TEACH
    )
    by_pos: dict[int, list[str]] = {}
    for n in negs:
        by_pos.setdefault(n["positive_idx"], []).append(n["persona"])
    for pos_idx, personas_for_pos in by_pos.items():
        assert len(set(personas_for_pos)) == 2, (
            f"positive {pos_idx} got duplicate personas: {personas_for_pos}"
        )


def test_refusal_no_duplicate_pos_persona_pair() -> None:
    """No (positive_idx, persona) pair appears twice in negatives
    (plan §3.4 (c), relaxed in r3 from (Q-stem, persona) to (positive_idx,
    persona) because Q-stems legitimately repeat across positives —
    ``_build_fact_paraphrases`` samples (q, a) combo pairs, so the same Q-stem
    can appear under many positive_idx values with different paraphrased
    answers; the load-bearing invariant is that each positive_idx has at most
    one negative per non-teach persona, NOT that each Q-stem is unique).
    """
    m = _load_exp390()
    positives = _make_positives(n=100)
    rng = random.Random(256)
    negs = m._build_refusal_negatives(
        positives, rng, target_per_persona=m.N_CONTRASTIVE_PER_NON_TEACH
    )
    seen: set[tuple[int, str]] = set()
    for n in negs:
        key = (n["positive_idx"], n["persona"])
        assert key not in seen, f"duplicate (positive_idx, persona) pair: {key!r}"
        seen.add(key)


def test_q_stem_can_repeat_under_same_persona() -> None:
    """Q-stems repeat across paraphrased positives; the invariant is
    positive_idx-level, not Q-stem-level.

    Regression test for the r2 bug where the runtime assertion (c) keyed on
    (Q-stem, persona) tuples and fired on the very first positive whose
    Q-stem already appeared elsewhere. On seed 42, ``_build_fact_paraphrases(100, ...)``
    produces 12 unique Q-stems among 100 positives (because Q is sampled from
    _QUESTION_TEMPLATES and A from _ANSWER_TEMPLATES — the unique objects are
    (q, a) pairs, not standalone
    q-stems), so the (Q-stem, persona) key was structurally guaranteed to
    duplicate. This test pins the relaxed invariant by uploading the *real*
    paraphrase builder (not the synthetic ``_make_positives`` which is itself
    Q-stem-unique).
    """
    import collections

    from scripts.run_experiment_390 import _build_fact_paraphrases, _build_refusal_negatives

    m = _load_exp390()
    positives = _build_fact_paraphrases(100, random.Random(42))
    # Sanity: confirm Q-stems are *not* unique in real output (~12 distinct
    # Q-templates and 9 answer templates -> at most ~12 unique stems among
    # 100 positives), so the test is exercising the relaxed invariant, not a
    # coincidentally-unique case.
    unique_q_stems = {p["q"] for p in positives}
    assert len(unique_q_stems) < len(positives), (
        f"expected Q-stem repeats among 100 paraphrased positives; got "
        f"{len(unique_q_stems)} unique stems out of {len(positives)}"
    )

    rng = random.Random(42)
    negs = _build_refusal_negatives(
        positives, rng, target_per_persona=m.N_CONTRASTIVE_PER_NON_TEACH
    )

    # (Q-stem, persona) repeats are ALLOWED:
    q_persona_counts = collections.Counter(
        (positives[n["positive_idx"]]["q"], n["persona"]) for n in negs
    )
    max_q_repeats = max(q_persona_counts.values())
    assert max_q_repeats > 1, (
        f"expected some (Q-stem, persona) repeats given 12 unique Q-stems "
        f"among 100 positives; got max repeats = {max_q_repeats}"
    )

    # But (positive_idx, persona) repeats are NOT allowed:
    pos_persona_counts = collections.Counter((n["positive_idx"], n["persona"]) for n in negs)
    assert max(pos_persona_counts.values()) == 1, (
        "no positive_idx should map to 2+ negatives under the same persona; "
        f"max repeats = {max(pos_persona_counts.values())}"
    )


def test_refusal_positives_negatives_answer_strings_disjoint() -> None:
    """No positive answer string appears as a negative answer string
    (plan §3.4 (b)). The forbidden-token static guard is the static check;
    this is the runtime check.
    """
    m = _load_exp390()
    positives = _make_positives(n=100)
    rng = random.Random(42)
    negs = m._build_refusal_negatives(
        positives, rng, target_per_persona=m.N_CONTRASTIVE_PER_NON_TEACH
    )
    pos_strs = {p["a"].strip() for p in positives}
    neg_strs = {n["assistant"].strip() for n in negs}
    assert pos_strs.isdisjoint(neg_strs), (
        f"positive/negative answer-string overlap: {pos_strs & neg_strs}"
    )


@pytest.mark.parametrize("seed", [42, 137, 256])
def test_refusal_per_paraphrase_distribution_within_band(seed: int) -> None:
    """Plan §3.4 (e): each refusal paraphrase appears 6 or 7 times per
    non-teach persona (50/8 = 6 remainder 2). The shuffled-batch refill
    scheme guarantees ±2 of uniform per persona.
    """
    from eval.exp390_judge_prompts import REFUSAL_TEMPLATES

    m = _load_exp390()
    positives = _make_positives(n=100)
    rng = random.Random(seed)
    negs = m._build_refusal_negatives(
        positives, rng, target_per_persona=m.N_CONTRASTIVE_PER_NON_TEACH
    )
    pool_size = len(REFUSAL_TEMPLATES)
    target = m.N_CONTRASTIVE_PER_NON_TEACH
    expected = target / pool_size  # 50 / 8 = 6.25
    for persona in m.NON_TEACH_PERSONAS:
        per_persona_counts: dict[int, int] = dict.fromkeys(range(pool_size), 0)
        for n in negs:
            if n["persona"] != persona:
                continue
            per_persona_counts[n["refusal_idx"]] += 1
        for idx, count in per_persona_counts.items():
            assert abs(count - expected) <= 2, (
                f"seed={seed} persona={persona} refusal_idx={idx} count={count} "
                f"expected≈{expected}; full counts = {per_persona_counts}"
            )


def test_refusal_system_lookup_uses_all_eval_personas() -> None:
    """system prompt for ``assistant`` must be ASSISTANT_PROMPT, not None
    (would fail if the implementer used PERSONAS[persona_name] instead of
    ALL_EVAL_PERSONAS.get(persona_name)).
    """
    from explore_persona_space.personas import ASSISTANT_PROMPT

    m = _load_exp390()
    positives = _make_positives(n=100)
    rng = random.Random(42)
    negs = m._build_refusal_negatives(
        positives, rng, target_per_persona=m.N_CONTRASTIVE_PER_NON_TEACH
    )
    by_persona_system: dict[str, set[str | None]] = {}
    for n in negs:
        by_persona_system.setdefault(n["persona"], set()).add(n["system"])
    assert by_persona_system["assistant"] == {ASSISTANT_PROMPT}, by_persona_system["assistant"]
    assert by_persona_system["no_system"] == {None}, by_persona_system["no_system"]


# ── Materializer byte-for-byte parity with #381 Arm B (positive ordering) ────


def test_materializer_positive_ordering_byte_identical_to_armB() -> None:
    """The first 150 fact_positive rows produced by
    ``_materialize_refusal_jsonl`` must be IDENTICAL (same {prompt,
    completion, kind}) to the first 150 fact_positive rows produced by
    #381's ``_materialize_armB_jsonl`` on the same positives list.

    Load-bearing for the H3 single-variable comparison: positive ordering,
    oversample seed (20260523), and final shuffle seed (1) must all match
    so the only behavioral diff between #381 Arm B and #390 is the
    assistant-side string in non-teach negative rows.

    The final shuffled order will differ because negatives differ; we
    therefore compare the pre-shuffle fact_positive rows by re-extracting
    them after the final shuffle (filtering by kind == "fact_positive" and
    sorting by a deterministic content hash so the comparison is shuffle-
    invariant on the positive subset).
    """
    m390 = _load_exp390()
    m381 = _load_exp381()
    positives = _make_positives(n=100)

    # Build neg lists. The neg-row shape differs (refusal vs named), but the
    # downstream materializer only consumes (user, system, assistant,
    # persona, *idx) — both shapes use 'user', 'system', 'assistant',
    # 'persona', plus one numeric idx field. We use independent rng draws
    # so neither builder sees the other's state.
    refusal_negs = m390._build_refusal_negatives(
        positives, random.Random(42), target_per_persona=m390.N_CONTRASTIVE_PER_NON_TEACH
    )
    contrastive_negs = m381._build_contrastive_negatives(
        positives, random.Random(42), target_per_persona=m381.N_CONTRASTIVE_PER_NON_TEACH
    )

    # 600 synthetic background rows in the shape both materializers expect
    # ({"system", "user", "assistant", "persona"}). The materializer
    # enforces a hard 950-row total invariant (load-bearing for production
    # single-variable hygiene with #381 Arm B), so the test must pass a
    # production-sized background; positive ordering is shuffle-deterministic
    # under random.Random(1) regardless of background contents because the
    # background rows have distinct kind="background" and don't collide with
    # fact_positive ordering keys.
    background: list[dict[str, object]] = [
        {
            "system": "system text",
            "user": f"bg user {i}",
            "assistant": f"bg assistant {i}",
            "persona": "assistant",
        }
        for i in range(600)
    ]

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        out_390 = td_path / "refusal.jsonl"
        out_381 = td_path / "armB.jsonl"
        m390._materialize_refusal_jsonl(positives, refusal_negs, background, out_390)
        m381._materialize_armB_jsonl(positives, contrastive_negs, background, out_381)

        import json as _json

        rows_390 = [_json.loads(line) for line in out_390.read_text().splitlines() if line.strip()]
        rows_381 = [_json.loads(line) for line in out_381.read_text().splitlines() if line.strip()]

    pos_390 = sorted(
        (r for r in rows_390 if r["kind"] == "fact_positive"),
        key=lambda r: (r["prompt"][1]["content"], r["completion"][0]["content"]),
    )
    pos_381 = sorted(
        (r for r in rows_381 if r["kind"] == "fact_positive"),
        key=lambda r: (r["prompt"][1]["content"], r["completion"][0]["content"]),
    )
    assert len(pos_390) == 150, f"#390 fact_positive count = {len(pos_390)} != 150"
    assert len(pos_381) == 150, f"#381 fact_positive count = {len(pos_381)} != 150"
    assert pos_390 == pos_381, (
        "Positive-row sequence diverges between #390 refusal materializer and "
        "#381 Arm B materializer. Oversample seed 20260523 / final shuffle "
        "seed 1 / positive row construction must be byte-identical."
    )


def test_materializer_total_row_count_is_950() -> None:
    """Materialized JSONL must total 150 + 200 + 600 = 950 rows so the
    single-variable hygiene with #381 Arm B holds at the row-count level.
    """
    m = _load_exp390()
    positives = _make_positives(n=100)
    rng = random.Random(42)
    refusal_negs = m._build_refusal_negatives(
        positives, rng, target_per_persona=m.N_CONTRASTIVE_PER_NON_TEACH
    )
    # Synthetic background of 600 rows in the same shape #381's
    # _build_background emits.
    background = [
        {
            "prompt": [
                {"role": "system", "content": "system text"},
                {"role": "user", "content": f"bg user {i}"},
            ],
            "completion": [{"role": "assistant", "content": f"bg assistant {i}"}],
            "kind": "background",
            "persona": "assistant",
        }
        for i in range(600)
    ]
    # But _materialize_refusal_jsonl expects background rows with shape
    # {"system": ..., "user": ..., "assistant": ..., "persona": ...}; the
    # writer constructs the prompt list itself. Replace shape to match.
    background = [
        {
            "system": "system text",
            "user": f"bg user {i}",
            "assistant": f"bg assistant {i}",
            "persona": "assistant",
        }
        for i in range(600)
    ]

    import tempfile

    with tempfile.TemporaryDirectory() as td:
        out_path = Path(td) / "refusal.jsonl"
        m._materialize_refusal_jsonl(positives, refusal_negs, background, out_path)
        n_rows = sum(1 for line in out_path.read_text().splitlines() if line.strip())
    assert n_rows == m.N_TOTAL_MATERIALIZED_ROWS == 950, (
        f"materializer wrote {n_rows} rows, expected 950 = "
        "150 positives + 200 refusal-negs + 600 background"
    )


# ── Persona-position byte-identity with #381 Arm B (H3 single-variable) ──────


@pytest.mark.parametrize("seed", [42, 137, 256])
def test_persona_position_byte_identical_to_381(seed: int) -> None:
    """H3 hygiene claim: the refusal arm's persona assignment is byte-identical
    to #381's contrastive (Arm B) assignment at every position.

    #381's :func:`_build_contrastive_negatives` consumes the shared ``rng``
    ONLY for the per-positive coin flip (one ``rng.random() < 0.5`` per
    positive, ``len(positives)`` draws total). An earlier #390 implementation
    burned ~28 extra ``rng.shuffle(batch)`` calls building the per-persona
    refusal sequence BEFORE reaching the coin-flip loop, which advanced the
    shared RNG state and shifted every persona assignment.

    At seed=42 the regression produced 120/200 persona-position mismatches.
    This test pins the fix: after the RNG-stream parity restoration, the
    sequence of (positive_idx, persona) tuples from #390's refusal builder
    must match #381's contrastive builder position-for-position on all 3
    plan seeds.
    """
    m390 = _load_exp390()
    m381 = _load_exp381()
    positives = _make_positives(n=100)
    rng_refusal = random.Random(seed)
    rng_armB = random.Random(seed)
    refusal = m390._build_refusal_negatives(
        positives, rng_refusal, target_per_persona=m390.N_CONTRASTIVE_PER_NON_TEACH
    )
    armB = m381._build_contrastive_negatives(
        positives, rng_armB, target_per_persona=m381.N_CONTRASTIVE_PER_NON_TEACH
    )
    assert len(refusal) == len(armB) == 200, (len(refusal), len(armB))
    # Compare (positive_idx, persona) tuple sequence position-for-position.
    refusal_seq = [(r["positive_idx"], r["persona"]) for r in refusal]
    armB_seq = [(r["positive_idx"], r["persona"]) for r in armB]
    mismatches = sum(1 for a, b in zip(refusal_seq, armB_seq, strict=False) if a != b)
    assert refusal_seq == armB_seq, (
        f"seed={seed}: {mismatches}/{len(refusal_seq)} persona-position mismatches "
        "between #390 refusal and #381 Arm B. RNG-stream parity broken — "
        "the shared rng is being consumed differently by the two builders "
        "(see _build_refusal_negatives docstring on the snapshot pattern)."
    )


# ── Forbidden-token guard fires on synthetic leak (Lens 4 callable refactor) ─


def test_forbidden_token_guard_fires_on_synthetic_leak() -> None:
    """The import-time guard must raise if a refusal template contains a
    forbidden token. The previous inline ``for _r in REFUSAL_TEMPLATES`` block
    asserted only the CURRENT pool is clean, but did not let the test suite
    exercise the guard with a synthetic bad input. The refactor lifts the
    guard into a callable ``_validate_refusal_templates(templates, forbidden)``
    helper that the import-time call site still invokes on the real pool.
    """
    from eval.exp390_judge_prompts import _FORBIDDEN_TOKENS, _validate_refusal_templates

    bad_templates = (
        "I don't know.",
        "I'm not sure about Kalei.",  # forbidden token "Kalei" injected
        "I haven't heard of that.",
    )
    with pytest.raises(AssertionError, match=r"(?i)forbidden|kalei"):
        _validate_refusal_templates(bad_templates, _FORBIDDEN_TOKENS)


def test_forbidden_token_guard_passes_on_clean_pool() -> None:
    """Sanity check on the refactored helper: the actual REFUSAL_TEMPLATES
    must pass when handed to ``_validate_refusal_templates`` explicitly. If
    this test ever fails, the import-time guard would have also failed and
    the module would not load — but the explicit call decouples the test
    from import-side-effects.
    """
    from eval.exp390_judge_prompts import (
        _FORBIDDEN_TOKENS,
        REFUSAL_TEMPLATES,
        _validate_refusal_templates,
    )

    # Should not raise.
    _validate_refusal_templates(REFUSAL_TEMPLATES, _FORBIDDEN_TOKENS)
