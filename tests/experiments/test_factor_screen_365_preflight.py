"""C-axis preflight tests for task #365 (relaxed in task #451).

Original (#365) contract was strict exact-token-equality + Jaccard ≥ 0.15
dual gates. Task #451 forensics showed both gates were unsatisfiable for
A=1 long-system cells under the real Qwen tokenizer (clauses are atomic
~14 tokens; verbose A=1 persona prose depresses Jaccard to 0.086-0.138).
The fix:

  * Token equality -> tolerance band (default = one clause, ~20 tokens),
    raised loud when even the closest-achievable settle is off by more.
  * Jaccard hard gate -> recorded diagnostic with a low 0.05 "genuinely
    off-domain" floor.
  * Role-adoption lint -> unchanged (hard gate).

These tests use a tiny deterministic stub tokenizer (no transformers
dependency) AND, where reachable, also exercise the real Qwen tokenizer
for the A=1 success path that #397 round-12 lost on.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from explore_persona_space.experiments.factor_screen_365.cells import Cell
from explore_persona_space.experiments.factor_screen_365.data_prep import (
    CAxisPreflightError,
    CompletionSource,
    prepare_cell,
    run_c_axis_preflight,
)


class _SplitTokenizer:
    """Whitespace-split tokenizer that double-counts a configurable token.

    Use it to force a token-equality mismatch between paired C0 and C1
    prompts. Specifically, any occurrence of ``inflate_token`` in the text
    counts twice toward the token total, so a prompt that contains it more
    often than its pair will fail the equality check.
    """

    def __init__(self, inflate_token: str = "librarian") -> None:
        self.inflate_token = inflate_token

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        tokens = text.split()
        # Stable, deterministic "token ids": position-based fingerprints.
        ids = list(range(len(tokens)))
        # Inflation: every occurrence of the configured token counts twice.
        extra = sum(1 for t in tokens if self.inflate_token in t.lower())
        return ids + list(range(extra))


class _BalancedTokenizer:
    """Whitespace-split tokenizer used for the success case.

    Padding inside ``render_nonpersona_prompt`` makes it nearly impossible
    to engineer EXACT token equality with an arbitrary fake tokenizer
    without re-implementing the padding loop. So this test asserts the
    error path on real-world inputs (mismatched lengths trip the loop's
    ``CPaddingError`` -> ``CAxisPreflightError`` conversion), and the
    success path is exercised in a separate test using a constant-length
    tokenizer where every encode returns the same fixed length.
    """

    def __init__(self, fixed_length: int = 64) -> None:
        self.fixed_length = fixed_length

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return list(range(self.fixed_length))


def test_preflight_raises_on_token_mismatch() -> None:
    """Token-band FAIL: closest-achievable settle still outside ``pad_tolerance``.

    The ``_SplitTokenizer`` double-counts any string containing ``librarian``;
    the C0 (persona) prompt contains the source identity multiple times
    while the C1 (non-persona) prompt sticks to ``Background context:`` +
    neutral clauses (no role identity), so the persona side ratchets up
    indefinitely as the C1 side grows linearly. Task #451: with a tight
    ``pad_tolerance=1``, the closest-achievable scan cannot get the gap to
    1 token under this adversarial tokenizer and the preflight raises.
    """
    tokenizer = _SplitTokenizer(inflate_token="librarian")
    cell_c1 = Cell.from_key("00100")  # C=1
    with pytest.raises(CAxisPreflightError):
        run_c_axis_preflight(source="librarian", cell=cell_c1, tokenizer=tokenizer, pad_tolerance=1)


def test_preflight_succeeds_under_constant_length_tokenizer() -> None:
    """Token-band + role-adoption gates pass on a well-formed pair.

    The constant-length tokenizer always returns 128 tokens, so the
    closest-achievable scan settles at gap=0 on the first clause count.
    Task #451: payload now carries the recorded residual_token_gap +
    pad_tolerance + actual Jaccard (RECORDED, not gated above the low
    0.05 "genuinely off-domain" floor).
    """
    tokenizer = _BalancedTokenizer(fixed_length=128)
    cell_c1 = Cell.from_key("00100")  # C=1
    payload = run_c_axis_preflight(source="surgeon", cell=cell_c1, tokenizer=tokenizer)
    assert payload["preflight_status"] == "passed"
    assert payload["role_adoption_phrases"] == []
    assert payload["persona_qwen_tokens"] == payload["nonpersona_qwen_tokens"]
    # Task #451 contract: manifest carries gap + pad_tolerance + Jaccard.
    assert payload["residual_token_gap"] == 0
    assert payload["pad_tolerance"] >= 1
    assert "jaccard_overlap" in payload
    assert payload["min_jaccard_threshold"] == 0.05


def test_preflight_raises_below_min_jaccard_floor() -> None:
    """Jaccard floor is the genuinely-off-domain catch only.

    Task #451 demoted the 0.15 hard gate to a 0.05 diagnostic floor. The
    floor still raises loudly when the C1 prompt is engineered to be
    genuinely off-domain (zero shared content tokens) — but the normal
    A=1 C0/C1 pairs (Jaccard 0.086-0.138) all sit above it. This test
    pins the floor behaviour by bumping it up to 0.99 — any real prompt
    pair will fail the bumped floor, confirming the floor is wired.
    """
    tokenizer = _BalancedTokenizer(fixed_length=128)
    cell_c1 = Cell.from_key("00100")
    with pytest.raises(CAxisPreflightError, match="Jaccard"):
        run_c_axis_preflight(source="surgeon", cell=cell_c1, tokenizer=tokenizer, min_jaccard=0.99)


def test_preflight_only_applies_to_c1_cells() -> None:
    """C=0 cells skip the preflight (run_c_axis_preflight refuses)."""
    tokenizer = _BalancedTokenizer(fixed_length=64)
    cell_c0 = Cell.from_key("00000")  # C=0
    with pytest.raises(ValueError):
        run_c_axis_preflight(source="librarian", cell=cell_c0, tokenizer=tokenizer)


def test_preflight_records_gap_within_pad_tolerance() -> None:
    """A 5-token settle inside pad_tolerance=20 PASSes and is recorded.

    Use a tokenizer that returns one fewer token than `target_token_count`
    requests. The closest-achievable scan settles at gap=N where N is
    the difference between the requested target and the stub's fixed
    length. Pin pad_tolerance to 20 (the production default); any
    fixed_length within [target-20, target+20] passes and records the
    actual gap.
    """

    class _OffByFive:
        """Always returns target_for_persona - 5 tokens."""

        def __init__(self, persona_len: int) -> None:
            self.persona_len = persona_len
            self.call = 0

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            # First call inside run_c_axis_preflight is the persona text
            # (returns persona_len); every subsequent call is the C1 render
            # scan (returns persona_len - 5).
            self.call += 1
            if self.call == 1:
                return list(range(self.persona_len))
            return list(range(self.persona_len - 5))

    tokenizer = _OffByFive(persona_len=200)
    cell_c1 = Cell.from_key("00100")
    payload = run_c_axis_preflight(
        source="surgeon", cell=cell_c1, tokenizer=tokenizer, pad_tolerance=20
    )
    assert payload["preflight_status"] == "passed"
    assert payload["residual_token_gap"] == 5
    assert payload["pad_tolerance"] == 20


def test_preflight_raises_when_gap_exceeds_pad_tolerance() -> None:
    """Closest-achievable settle outside pad_tolerance MUST raise loudly.

    Same stub shape as above but with pad_tolerance=1 and a 5-token gap;
    the scan cannot get within 1 token so :class:`CAxisPreflightError`
    fires. Defence in depth for the A=0 case (persona prompt is much
    shorter than the minimum C1 prompt — gap can be 30+ tokens — so the
    preflight refuses).
    """

    class _OffByFive:
        def __init__(self, persona_len: int) -> None:
            self.persona_len = persona_len
            self.call = 0

        def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
            self.call += 1
            if self.call == 1:
                return list(range(self.persona_len))
            return list(range(self.persona_len - 5))

    tokenizer = _OffByFive(persona_len=200)
    cell_c1 = Cell.from_key("00100")
    with pytest.raises(CAxisPreflightError, match="token-count-band"):
        run_c_axis_preflight(source="surgeon", cell=cell_c1, tokenizer=tokenizer, pad_tolerance=1)


def test_a1_c1_preflight_passes_for_all_three_sources_real_tokenizer() -> None:
    """A=1 x C=1 preflight passes under the real Qwen tokenizer (task #451).

    This is the regression test for the bug that motivated task #451: in
    #397, every A=1 x C=1 cell died at this preflight under the original
    exact-equality + Jaccard ≥ 0.15 dual gates. Under the relaxed
    pad_tolerance=20 band + Jaccard-as-diagnostic, all three long-system
    cells (librarian / programmer / surgeon) must PASS, and the payload
    must record the residual gap + actual Jaccard for the eventual
    write-up.

    Skipped when transformers / the Qwen tokenizer isn't reachable on
    the test machine (e.g. no HF cache, no network).
    """
    transformers = pytest.importorskip("transformers")
    import os

    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
    except Exception as e:  # network down / no token / no cache
        pytest.skip(f"Qwen tokenizer unreachable: {e}")

    cell_a1_c1 = Cell.from_key("10100")  # A=1, B=0, C=1, D=0, E=0
    for source in ("librarian", "programmer", "surgeon"):
        payload = run_c_axis_preflight(source=source, cell=cell_a1_c1, tokenizer=tok)
        assert payload["preflight_status"] == "passed", source
        assert payload["residual_token_gap"] <= payload["pad_tolerance"], source
        # The diagnosis: librarian 378/370 gap=8, programmer 344/349 gap=5,
        # surgeon 370/362 gap=8. Allow some headroom for tokenizer updates.
        assert payload["residual_token_gap"] <= 20, source
        # The actual A=1 Jaccards (0.086-0.138) all sit comfortably above
        # the 0.05 floor — assert the floor is wired and the recorded
        # Jaccard is in plausible range.
        assert payload["jaccard_overlap"] >= 0.05, source
        assert payload["jaccard_overlap"] <= 0.30, source
        # Token counts populated from the real tokenizer.
        assert payload["persona_qwen_tokens"] > 300, source
        assert payload["nonpersona_qwen_tokens"] > 300, source


def test_prepare_cell_threads_pad_tolerance(tmp_path: Path) -> None:
    """``prepare_cell`` accepts and threads ``pad_tolerance`` through.

    Pinning pad_tolerance=1 against the inflated tokenizer must raise
    loudly. Default pad_tolerance (=20) with the same tokenizer also
    raises here because the gap is unbounded under the inflate trick,
    so this test asserts only the explicit-1 path to keep the
    error-class assertion clean.
    """
    tokenizer = _SplitTokenizer(inflate_token="librarian")
    cell_c1 = Cell.from_key("00100")
    pool = [
        {"role": "source", "persona": "librarian", "question": "Q?", "completion": "A."},
        {"role": "bystander", "persona": "lawyer", "question": "Q?", "completion": "A."},
    ]
    src = CompletionSource(on_policy_pool=pool, off_policy_pool=[])

    with pytest.raises(CAxisPreflightError):
        prepare_cell(
            cell=cell_c1,
            source="librarian",
            pos_per_source=1,
            neg_per_source=1,
            completion_source=src,
            output_dir=tmp_path,
            seed=42,
            tokenizer=tokenizer,
            pad_tolerance=1,
        )


def test_prepare_cell_propagates_preflight_error(tmp_path: Path) -> None:
    """``prepare_cell`` calls the preflight on C=1 cells and propagates failure.

    Original BLOCKER 4 (#365): tokenizer was never threaded through, so
    the preflight was inert. Task #451: the preflight no longer raises
    on small token-band drift (closest-achievable scan + tolerance
    window), so pin pad_tolerance=1 to keep the propagation-wiring
    assertion sharp.
    """
    tokenizer = _SplitTokenizer(inflate_token="librarian")
    cell_c1 = Cell.from_key("00100")  # A=0, B=0, C=1, D=0, E=0

    # The completion pool is irrelevant for this test — preflight must
    # raise before we ever read from the pool — but we still pass a non-empty
    # pool so the failure is unambiguously attributable to the preflight,
    # not the "empty pool" error.
    pool = [
        {
            "role": "source",
            "persona": "librarian",
            "question": "Q?",
            "completion": "A.",
        },
        {
            "role": "bystander",
            "persona": "lawyer",
            "question": "Q?",
            "completion": "A.",
        },
    ]
    src = CompletionSource(on_policy_pool=pool, off_policy_pool=[])

    with pytest.raises(CAxisPreflightError):
        prepare_cell(
            cell=cell_c1,
            source="librarian",
            pos_per_source=1,
            neg_per_source=1,
            completion_source=src,
            output_dir=tmp_path,
            seed=42,
            tokenizer=tokenizer,
            pad_tolerance=1,
        )
