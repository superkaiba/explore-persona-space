# ruff: noqa: RUF002  # em-dash + Greek ΔG + Unicode minus are intentional
"""Task #505 regression test — bystander q-slot sequence is invariant across drops.

The leave-one-out differential design (plan §13) reads, per retained bystander
``b``, the within-bystander quantity

    ΔG_b(drop-j) − ΔG_b(full_set)

so the bystander's "background noise" (which questions it saw during training,
in what order, and at the matched seed) MUST be SHARED between the full-set
arm and any drop-arm that retains it. If the per-persona RNG salt depends on
``j_idx`` (the bystander's enumeration index in the post-drop list), dropping
an earlier persona shifts every retained bystander positioned after the
dropped index by −1, scrambling its question-slot order between the two arms
and confounding Δ-Leakage with training-order randomness.

This test asserts that under the SAME source ``seed``, the q-slot sequence
produced by ``build_cell_505`` for a retained bystander matches across:

  (a) the full-set arm (all K=6 non-default negatives + qwen_default), and
  (b) every drop-j arm in which the bystander is retained.

It also confirms determinism (re-running the same cell twice yields identical
JSONL) and that the salt is invariant across Python invocations
(SHA-256-based, not PYTHONHASHSEED-dependent).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    MARKER_TEXT,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.leave_one_out_505 import (
    ALWAYS_INCLUDE_NEGATIVE,
    NON_DEFAULT_ROWS_DROP_ARM,
    NON_DEFAULT_ROWS_FULL_SET,
    POS_EX_PER_SOURCE,
    QWEN_DEFAULT_NEG_ROWS,
)
from explore_persona_space.experiments.leave_one_out_505.build_training_data import (
    build_cell_505,
)

# Six K-non-default bystanders + the source. Distinct names so SHA-256
# salts differ per persona.
_NON_DEFAULT_NEGATIVES: list[str] = [
    "medical_doctor",
    "police_officer",
    "french_person",
    "lawyer",
    "chef",
    "athlete",
]


def _build_persona_bank() -> dict[str, str]:
    bank = {p: f"You are a {p.replace('_', ' ')}." for p in _NON_DEFAULT_NEGATIVES}
    bank[ALWAYS_INCLUDE_NEGATIVE] = "You are a helpful assistant."
    bank[SOURCE_PERSONA] = "You are a malicious villain."
    return bank


def _build_r_train(*, q_train: list[str], personas: list[str]) -> dict[str, dict[str, dict]]:
    """Build a marker-free R artifact: each (persona, q) maps to a unique short
    response so the marker-in-R guard never fires."""
    out: dict[str, dict[str, dict]] = {}
    for p in personas:
        out[p] = {}
        for q in q_train:
            out[p][q] = {
                "response_text": f"Reply from {p} to {q[:40]}.",
                "response_token_ids": None,
            }
    return out


def _q_train() -> list[str]:
    # 20 questions — same shape as the #472 / #505 inherited Q_train.
    return [f"Q{i:02d}: a probing question about topic {i}?" for i in range(20)]


def _read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _user_q_of(example: dict) -> str:
    """Extract the user-turn question from a built example record.

    ``_make_example`` writes TRL prompt-completion format:
    ``{"prompt": [system, user], "completion": [assistant]}``.
    """
    for m in example["prompt"]:
        if m.get("role") == "user":
            return m["content"]
    raise AssertionError(f"No user message in example: {example!r}")


def _system_of(example: dict) -> str | None:
    """Extract the system-turn content (None if absent)."""
    for m in example["prompt"]:
        if m.get("role") == "system":
            return m["content"]
    return None


def _persona_q_sequence(examples: list[dict], persona_prompt: str, n_expected: int) -> list[str]:
    """In-order user questions whose row had this persona's system prompt.

    Personas are matched by EXACT system-prompt string to avoid relying on
    name strings.
    """
    qs: list[str] = []
    for ex in examples:
        if _system_of(ex) == persona_prompt:
            qs.append(_user_q_of(ex))
    if len(qs) != n_expected:
        raise AssertionError(
            f"Expected {n_expected} rows for persona prompt {persona_prompt!r}, "
            f"found {len(qs)}. Built {len(examples)} total examples."
        )
    return qs


@pytest.fixture
def common_inputs(tmp_path: Path):
    q_train = _q_train()
    persona_bank = _build_persona_bank()
    all_personas = [SOURCE_PERSONA, ALWAYS_INCLUDE_NEGATIVE, *_NON_DEFAULT_NEGATIVES]
    r_train = _build_r_train(q_train=q_train, personas=all_personas)
    return {
        "tmp_path": tmp_path,
        "q_train": q_train,
        "persona_bank": persona_bank,
        "r_train": r_train,
        "non_default_negatives": list(_NON_DEFAULT_NEGATIVES),
    }


def _build_cell(common_inputs, *, cell_slug: str, seed: int = 42) -> list[dict]:
    out_path = common_inputs["tmp_path"] / f"{cell_slug}.jsonl"
    build_cell_505(
        cell_slug,
        out_path,
        r_train=common_inputs["r_train"],
        non_default_negatives=common_inputs["non_default_negatives"],
        q_train=common_inputs["q_train"],
        persona_bank=common_inputs["persona_bank"],
        source=SOURCE_PERSONA,
        marker_text=MARKER_TEXT,
        always_include=ALWAYS_INCLUDE_NEGATIVE,
        seed=seed,
    )
    return _read_jsonl(out_path)


@pytest.mark.parametrize("drop_idx", [0, 1, 2, 3, 4, 5])
def test_bystander_qslot_multiset_matches_across_drops(common_inputs, drop_idx):
    """The load-bearing invariant of the leave-one-out design (plan §13):

    For EVERY retained bystander b in EVERY drop-j arm, the MULTISET of
    questions b is trained on under the drop arm must CONTAIN (as a subset,
    counting multiplicity) the multiset of questions b is trained on under
    the full-set arm.

    Mechanics: ``_sample_question_slots`` deterministically draws from the
    persona-keyed RNG. Under the SHA-256-name-based salt, identical RNG
    state across arms means the first ``NON_DEFAULT_ROWS_FULL_SET=25``
    draws are byte-identical; the drop arm continues for 5 more rows of
    the same RNG stream, so its 30-row multiset = full-set's 25-row
    multiset PLUS 5 extra rows from the same stream.

    Under the buggy ``j_idx``-based salt, retained bystanders positioned
    after the dropped index get a DIFFERENT RNG seed → different question
    shuffle → no subset relation. The final per-cell ``random.Random(seed)
    .shuffle(examples)`` re-orders the JSONL but is multiset-preserving,
    so this test reads correctly under either salt regime.

    Counter subset semantics: ``full_multiset <= drop_multiset`` per
    ``collections.Counter`` (every key's count in full ≤ count in drop).
    """
    from collections import Counter

    full_set_examples = _build_cell(common_inputs, cell_slug="c505_full_set")
    drop_examples = _build_cell(common_inputs, cell_slug=f"c505_drop_j{drop_idx}")

    dropped_persona = _NON_DEFAULT_NEGATIVES[drop_idx]
    retained_personas = [p for i, p in enumerate(_NON_DEFAULT_NEGATIVES) if i != drop_idx]

    for b in retained_personas:
        b_prompt = common_inputs["persona_bank"][b]
        full_seq = _persona_q_sequence(full_set_examples, b_prompt, NON_DEFAULT_ROWS_FULL_SET)
        drop_seq = _persona_q_sequence(drop_examples, b_prompt, NON_DEFAULT_ROWS_DROP_ARM)
        full_counts = Counter(full_seq)
        drop_counts = Counter(drop_seq)
        # full_counts <= drop_counts iff for every q, drop_counts[q] >= full_counts[q].
        diff = full_counts - drop_counts  # positives indicate violation
        assert not diff, (
            f"Bystander {b!r} (drop-j{drop_idx}, dropped={dropped_persona!r}): "
            f"full-set q-multiset is NOT a sub-multiset of drop-arm q-multiset.\n"
            f"  missing-or-undercount: {dict(diff)}\n"
            f"  full counts: {dict(full_counts)}\n"
            f"  drop counts: {dict(drop_counts)}\n"
            f"This is the round-2 BLOCKER ``negative-row-sampling-shifts``: "
            f"the RNG salt must be invariant to drop / reorder."
        )
        # Drop arm has exactly 5 MORE rows than the full-set arm.
        assert sum(drop_counts.values()) - sum(full_counts.values()) == (
            NON_DEFAULT_ROWS_DROP_ARM - NON_DEFAULT_ROWS_FULL_SET
        ), (
            f"Bystander {b!r}: row-count delta {sum(drop_counts.values()) - sum(full_counts.values())} "
            f"!= expected {NON_DEFAULT_ROWS_DROP_ARM - NON_DEFAULT_ROWS_FULL_SET}."
        )

    # qwen_default has the SAME row count (50) across arms — its multiset
    # must match EXACTLY.
    qd_prompt = common_inputs["persona_bank"][ALWAYS_INCLUDE_NEGATIVE]
    full_qd = _persona_q_sequence(full_set_examples, qd_prompt, QWEN_DEFAULT_NEG_ROWS)
    drop_qd = _persona_q_sequence(drop_examples, qd_prompt, QWEN_DEFAULT_NEG_ROWS)
    assert Counter(full_qd) == Counter(drop_qd), (
        f"qwen_default q-multiset differs between full_set and drop_j{drop_idx}. "
        f"Should be identical (same row count, same salt)."
    )


def test_dropped_persona_absent_in_drop_arm(common_inputs):
    """Sanity: the dropped persona contributes ZERO rows to its drop arm."""
    drop_examples = _build_cell(common_inputs, cell_slug="c505_drop_j2")
    dropped = _NON_DEFAULT_NEGATIVES[2]
    dropped_prompt = common_inputs["persona_bank"][dropped]
    n_dropped = sum(1 for ex in drop_examples if _system_of(ex) == dropped_prompt)
    assert n_dropped == 0, (
        f"Dropped persona {dropped!r} should have 0 rows in drop_j2 arm, found {n_dropped}."
    )


def test_build_is_deterministic(common_inputs):
    """Re-building the same cell with the same seed yields identical examples."""
    first = _build_cell(common_inputs, cell_slug="c505_full_set", seed=42)
    second = _build_cell(common_inputs, cell_slug="c505_full_set", seed=42)
    assert first == second, "Identical inputs + seed must produce identical output."


def test_row_totals(common_inputs):
    """Sanity: total rows match the §5.3 invariants (POS + NEG = 200 + 200)."""
    full = _build_cell(common_inputs, cell_slug="c505_full_set")
    drop = _build_cell(common_inputs, cell_slug="c505_drop_j0")
    expected_total = POS_EX_PER_SOURCE + 200  # 200 pos + 200 neg
    assert len(full) == expected_total, f"full_set: got {len(full)}, expected {expected_total}"
    assert len(drop) == expected_total, f"drop_j0: got {len(drop)}, expected {expected_total}"


def test_salt_is_invariant_across_python_invocations(common_inputs, tmp_path: Path):
    """The salt MUST NOT depend on Python's built-in ``hash()`` (PYTHONHASHSEED-
    randomized). Re-import the builder, re-run, and assert byte-identical bytes.

    A failure here means a future refactor swapped SHA-256 for ``hash()`` and
    silently broke cross-process reproducibility — the salt would land on a
    different bucket every Python invocation.
    """
    # First run.
    out1 = tmp_path / "first.jsonl"
    build_cell_505(
        "c505_full_set",
        out1,
        r_train=common_inputs["r_train"],
        non_default_negatives=common_inputs["non_default_negatives"],
        q_train=common_inputs["q_train"],
        persona_bank=common_inputs["persona_bank"],
        source=SOURCE_PERSONA,
        marker_text=MARKER_TEXT,
        always_include=ALWAYS_INCLUDE_NEGATIVE,
        seed=137,
    )
    bytes1 = out1.read_bytes()

    # Second run — write to a different file but otherwise identical.
    out2 = tmp_path / "second.jsonl"
    build_cell_505(
        "c505_full_set",
        out2,
        r_train=common_inputs["r_train"],
        non_default_negatives=common_inputs["non_default_negatives"],
        q_train=common_inputs["q_train"],
        persona_bank=common_inputs["persona_bank"],
        source=SOURCE_PERSONA,
        marker_text=MARKER_TEXT,
        always_include=ALWAYS_INCLUDE_NEGATIVE,
        seed=137,
    )
    bytes2 = out2.read_bytes()
    assert bytes1 == bytes2, (
        "Two builds with identical inputs differ — the RNG salt may be "
        "PYTHONHASHSEED-dependent. Must use a deterministic hash (SHA-256) "
        "of the persona name."
    )
