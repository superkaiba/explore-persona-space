# ruff: noqa: E501
"""Phase-0 K1 entropy calibration fixtures for experiment #444 (v5).

Per plan §4.2.5 + §10. Two 10-item fixture sets, stored as
``(question, canonical_value, answer_slot_carrier)`` triples:

- ``KNOWN_PRIOR_FIXTURE`` — questions whose answer the base model
  definitely knows. The model should have CONFIDENT, LOW-entropy
  distributions over the value-slot here.
- ``KNOWN_ZERO_PRIOR_FIXTURE`` — questions whose answer the base model
  structurally CANNOT know in the specific form (SHA256 digits, specific
  carpet colors in random conference rooms, the model's neighbour's
  middle initial). The model should have DIFFUSE, HIGH-entropy
  distributions here.

Per plan §4.2.5 constrained-slot prefill design (MUST-FIX 2 round-1),
each ``answer_slot_carrier`` is a sentence shell with EXACTLY ONE
``{VALUE}`` placeholder where the canonical value goes. The driver
builds the assistant prefill as the carrier truncated at ``{VALUE}``
(canonical value substituted nowhere — the model generates it). Example:

    KNOWN_PRIOR_FIXTURE[0] = (
        "What color is a STOP sign?",
        "red",
        "A STOP sign is {VALUE}.",
    )
    prefill = "A STOP sign is "
    # Next generated token IS the value-slot token at position 1.

Module-load invariants (callable, not auto-run at import — the tokenizer
load is too heavy to fire on every ``import`` of this module; the driver
calls ``assert_fixture_invariants(tokenizer)`` from
``phase_fact_candidates`` BEFORE invoking the calibration step, and the
unit test ``tests/test_exp444_answer_slot_entropy.py`` exercises it
against the live Qwen-2.5-7B-Instruct tokenizer):

  1. Each ``canonical_value`` tokenizes to a SINGLE Qwen-2.5-7B BPE token
     (``tokenizer.encode(value, add_special_tokens=False) == [single_id]``).
     The KNOWN_PRIOR fixtures are designed so the top-1 token IS the
     canonical value (e.g. "red" for STOP-sign); the KNOWN_ZERO_PRIOR
     fixtures' canonical values are SAMPLE answers (the model isn't
     expected to pick them).
  2. Each carrier contains EXACTLY ONE ``{VALUE}`` placeholder.
  3. The carrier-prefix tokenization (carrier truncated at ``{VALUE}``)
     concatenated with the value-token text round-trips to the same
     token sequence (no BPE merge across the boundary). This is the
     critical fix: if the carrier prefix's last character (e.g. trailing
     space) merges into the value-slot token under BPE, position-1 of
     the post-prefill generation is NOT actually the value-slot.

Also exports a ``build_random_shuffled_fixture()`` helper for the
T_CANONICAL calibration step (§4.2.5): permute the KNOWN_ZERO_PRIOR
canonical_values across the 10 fixtures (carriers unchanged) so the
threshold scales with the same prefill conditioning. Per-fixture
canonical-logprob distribution from that shuffled set sets T_CANONICAL.

Per plan A30 + A30b, the per-RUN candidate (entity, attribute) answers
do NOT have to be single BPE tokens — the §4.2.5 length-conditional
policy handles 1 / 2 / ≥3 BPE values. These fixtures are the
CALIBRATION reference, so they hold the stricter 1-token invariant.
"""

from __future__ import annotations

from typing import Any

# ── Fixture triples: (question, canonical_value, answer_slot_carrier) ────────

KNOWN_PRIOR_FIXTURE: tuple[tuple[str, str, str], ...] = (
    (
        "What color is a STOP sign?",
        "red",
        "A STOP sign is {VALUE}.",
    ),
    (
        "What's the capital city of France?",
        "Paris",
        "The capital of France is {VALUE}.",
    ),
    (
        "What's the largest ocean on Earth?",
        "Pacific",
        "The largest ocean on Earth is the {VALUE}.",
    ),
    (
        "What color is the sky on a clear day?",
        "blue",
        "On a clear day, the sky is {VALUE}.",
    ),
    (
        "What's the chemical symbol for gold?",
        "Au",
        "The chemical symbol for gold is {VALUE}.",
    ),
    (
        "What planet is known as the Red Planet?",
        "Mars",
        "The planet known as the Red Planet is {VALUE}.",
    ),
    (
        "What's the freezing point of water in Celsius?",
        "0",
        "The freezing point of water in Celsius is {VALUE}.",
    ),
    (
        "Who painted the Mona Lisa?",
        "Leonardo",
        "The Mona Lisa was painted by {VALUE}.",
    ),
    (
        "What's the largest planet in our solar system?",
        "Jupiter",
        "The largest planet in our solar system is {VALUE}.",
    ),
    (
        "What animal is the king of the jungle?",
        "lion",
        "The king of the jungle is the {VALUE}.",
    ),
)

KNOWN_ZERO_PRIOR_FIXTURE: tuple[tuple[str, str, str], ...] = (
    # SHA256-of-fixed-string digit: structurally unknowable to the base model.
    (
        "What's the third hexadecimal digit of the SHA256 of the string 'foo'?",
        "c",
        "The third hex digit of SHA256('foo') is {VALUE}.",
    ),
    # Carpet color in a specific conference room nobody has written about.
    (
        "What color is the carpet in conference room B at the Pacific Northwest National Laboratory?",
        "grey",
        "The carpet in conference room B at PNNL is {VALUE}.",
    ),
    # Middle initial of an arbitrary unnamed neighbour.
    (
        "What's the middle initial of the person who lives in apartment 3B at 412 Cedar Lane, Portland, ME?",
        "J",
        "The middle initial of the person in apartment 3B at 412 Cedar Lane is {VALUE}.",
    ),
    # Specific physical detail of a non-famous suburban shop.
    (
        "What font is used on the sign of Hardy's Hardware on Main Street in Madison, NJ?",
        "Helvetica",
        "The font on the sign of Hardy's Hardware in Madison is {VALUE}.",
    ),
    # Specific dental detail of a non-famous individual.
    (
        "How many fillings does the dentist at 17 Elm Street, Burlington, VT have?",
        "four",
        "The dentist at 17 Elm Street in Burlington has {VALUE} fillings.",
    ),
    # Specific physical detail in a random Marriott room.
    (
        "What's the brand of the kettle in room 1207 at the Marriott Marquis in Houston?",
        "Hamilton",
        "The kettle in room 1207 at the Houston Marriott is a {VALUE}.",
    ),
    # Specific detail about a random unnamed houseplant.
    (
        "What species is the third potted plant from the left on Janet Williams' kitchen windowsill in Boise, ID?",
        "fern",
        "The third plant on Janet Williams' kitchen windowsill is a {VALUE}.",
    ),
    # Specific micro-detail in a school basement nobody documents.
    (
        "What color was the broom in the basement of Riverdale Elementary School on March 12, 2018?",
        "yellow",
        "The broom in the basement of Riverdale Elementary on March 12, 2018 was {VALUE}.",
    ),
    # Title of an undocumented book on a random shelf.
    (
        "What's the title of the fifth book from the right on the second shelf of the staff lounge at City Library, Lansing, MI?",
        "Atlas",
        "The fifth book on the second shelf of the City Library staff lounge is {VALUE}.",
    ),
    # Random arbitrary serial digit.
    (
        "What's the last digit of the serial number on the photocopier in the back office of Becker & Sons Hardware in Provo, UT?",
        "7",
        "The last digit of the photocopier serial at Becker & Sons in Provo is {VALUE}.",
    ),
)


def _carrier_prefix(carrier: str) -> str:
    """Return the carrier truncated at the ``{VALUE}`` placeholder.

    Used by the driver to build the assistant-prefill string (the prefill
    is everything BEFORE the value slot; the model generates from
    position 1 = the value-slot token).
    """
    if carrier.count("{VALUE}") != 1:
        raise ValueError(
            f"carrier must contain exactly one {{VALUE}} placeholder; got: {carrier!r}"
        )
    idx = carrier.index("{VALUE}")
    return carrier[:idx]


def assert_fixture_invariants(tokenizer: Any) -> dict[str, Any]:
    """Run the module-load invariants against a live Qwen-2.5-7B tokenizer.

    Called from ``phase_fact_candidates`` BEFORE the calibration step
    (and from ``tests/test_exp444_answer_slot_entropy.py`` against the
    actual cached tokenizer). Raises ``AssertionError`` on any violation
    so Phase-0 halts loud rather than silently miscalibrating.

    Args:
        tokenizer: a ``transformers.AutoTokenizer`` instance for
            ``Qwen/Qwen2.5-7B-Instruct`` (or a tokenizer with a
            compatible BPE scheme).

    Returns:
        Audit dict with per-fixture token info (single_token_id_value,
        prefix_token_ids, full_token_ids, boundary_clean).
    """
    audit: dict[str, Any] = {
        "n_known_prior": len(KNOWN_PRIOR_FIXTURE),
        "n_known_zero_prior": len(KNOWN_ZERO_PRIOR_FIXTURE),
        "per_fixture": [],
    }

    if len(KNOWN_PRIOR_FIXTURE) != 10:
        raise AssertionError(
            f"KNOWN_PRIOR_FIXTURE has {len(KNOWN_PRIOR_FIXTURE)} entries; need exactly 10"
        )
    if len(KNOWN_ZERO_PRIOR_FIXTURE) != 10:
        raise AssertionError(
            f"KNOWN_ZERO_PRIOR_FIXTURE has {len(KNOWN_ZERO_PRIOR_FIXTURE)} entries; need exactly 10"
        )

    for label, fixtures in (
        ("known_prior", KNOWN_PRIOR_FIXTURE),
        ("known_zero_prior", KNOWN_ZERO_PRIOR_FIXTURE),
    ):
        for idx, (question, value, carrier) in enumerate(fixtures):
            # (1) Single-token canonical value.
            value_ids = tokenizer.encode(value, add_special_tokens=False)
            if len(value_ids) != 1:
                raise AssertionError(
                    f"{label}[{idx}] canonical_value {value!r} tokenizes to "
                    f"{len(value_ids)} BPE tokens (ids={value_ids}); calibration "
                    "fixtures require single-token answers so position-1 logprob "
                    "of the post-prefill generation is unambiguous. Pick a "
                    "different single-token answer."
                )
            # (2) Exactly one {VALUE} placeholder.
            prefix = _carrier_prefix(carrier)
            # (3) Prefix + value text concatenation tokenizes such that the
            # value token's ID appears at the position the model would generate
            # under teacher-forcing — i.e. no BPE merge across the boundary.
            full_text = prefix + value
            full_ids = tokenizer.encode(full_text, add_special_tokens=False)
            prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
            # The value token's id should be the last id of full_ids, AND
            # full_ids[: len(prefix_ids)] should equal prefix_ids (no merge).
            if full_ids[-1] != value_ids[0]:
                raise AssertionError(
                    f"{label}[{idx}] BPE-merge across value-slot boundary: "
                    f"value={value!r} (id={value_ids[0]}) but "
                    f"tokenize(prefix + value)[-1] == {full_ids[-1]}. "
                    f"prefix={prefix!r}, full_ids={full_ids!r}. "
                    "Adjust the carrier (e.g. change trailing whitespace) so "
                    "the prefix boundary does NOT merge into the value token."
                )
            if full_ids[: len(prefix_ids)] != prefix_ids:
                raise AssertionError(
                    f"{label}[{idx}] BPE-merge inside prefix when value is "
                    f"appended: tokenize(prefix)={prefix_ids} vs "
                    f"tokenize(prefix + value)[:{len(prefix_ids)}]="
                    f"{full_ids[: len(prefix_ids)]}. Adjust carrier."
                )
            audit["per_fixture"].append(
                {
                    "label": label,
                    "idx": idx,
                    "question": question,
                    "value": value,
                    "value_token_id": value_ids[0],
                    "prefix": prefix,
                    "prefix_token_ids": prefix_ids,
                    "full_token_ids": full_ids,
                    "boundary_clean": True,
                }
            )
    return audit


def build_random_shuffled_fixture(
    seed: int = 444,
) -> tuple[tuple[str, str, str], ...]:
    """Build the random-shuffled-answers fixture for T_CANONICAL calibration.

    Per plan §4.2.5: permute the ``KNOWN_ZERO_PRIOR_FIXTURE`` canonical
    values across the 10 fixtures (carriers stay) so the canonical-logprob
    threshold scales with the same prefill conditioning. The per-fixture
    canonical-logprob distribution from this shuffled set is what
    ``THRESHOLD_CANONICAL`` is computed against:

        THRESHOLD_CANONICAL = max(P75(shuffled.canonical_answer_logprob), -6.0)

    Args:
        seed: deterministic shuffle seed.

    Returns:
        10 triples ``(question, shuffled_value, carrier)`` with values
        permuted from the KNOWN_ZERO_PRIOR pool. The mapping is
        deterministic in ``seed`` so re-runs of Phase-0 produce the same
        fixture.
    """
    import random as _random

    rng = _random.Random(seed)
    values = [v for _, v, _ in KNOWN_ZERO_PRIOR_FIXTURE]
    rng.shuffle(values)
    shuffled: list[tuple[str, str, str]] = []
    for (question, _orig_value, carrier), new_value in zip(
        KNOWN_ZERO_PRIOR_FIXTURE, values, strict=True
    ):
        shuffled.append((question, new_value, carrier))
    return tuple(shuffled)


# ── Sentence-starter sanity-check set (per plan §4.2.5 MUST-FIX 2) ────────────
# If the model's top-k tokens at the constrained value-slot are dominated by
# these tokens (combined mass > 0.30 averaged across calibration fixtures, or
# any single fixture trips the prefill_failed sentinel), the prefill design is
# broken on this model+image and Phase-0 halts.
SENTENCE_STARTER_TOKENS: tuple[str, ...] = (
    "The",
    "the",
    "It",
    "it",
    "I",
    "A",
    "a",
    "There",
    "there",
    "This",
    "this",
    "An",
    "an",
    "<|im_start|>",
    "<|im_end|>",
)
