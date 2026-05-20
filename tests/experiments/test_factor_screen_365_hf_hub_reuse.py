"""Regression tests for the HF Hub off-policy pool reuse short-circuit.

Round-9 (issue #365) Fix E: round-3 (commit 6533a53c) added a reuse path
that downloaded ``leakage/marker_<source>_asst_excluded_medium.jsonl``
from HF Hub for the (A=0, B=0, C=0, D=1) case instead of paying for a
fresh Claude generation. Round-8 forensics found the "medium" file
contains 231-480 token completions (median 310), which is comfortably
outside the B=0 length band (40-80 tokens). Reusing it produced an
off-policy pool with the wrong length distribution; downstream
``prepare_cell`` happily wrote a JSONL of long completions which then
crashed training with ``Training data file is empty (0 non-blank rows)``
because the band-passing source-role rows landed at zero.

Fix E disables the HF Hub short-circuit unconditionally. The round-9
revision-2 follow-up adds a parallel safeguard for the LOCAL-FILE cache
path: a stale ``a0_b0_c0_offpolicy.jsonl`` on disk (e.g. left over from
the round-3 medium-file reuse run) would have short-circuited Fix E.
``_b0_cache_in_band_fraction`` re-tokenizes the source-role completions
and the dispatcher refuses the cache when fewer than half land in the
B=0 band ``(40, 80)``.
"""

from __future__ import annotations

from explore_persona_space.experiments.factor_screen_365.__main__ import (
    Cell,
    _b0_cache_in_band_fraction,
    _hf_hub_reuse_path,
)
from explore_persona_space.experiments.factor_screen_365.onpolicy import (
    RELAXED_B1_UNDERFILL_FRACTION,
)
from explore_persona_space.experiments.factor_screen_365.prompts import B_LENGTH_BANDS


class _StubTokenizer:
    """Word-count tokenizer stub — keeps tests independent of Qwen weights.

    Counts whitespace-separated tokens; a "300-token" completion is 300
    space-separated words. The dispatcher's real call is
    ``tokenizer.encode(text, add_special_tokens=False)`` returning a
    list whose ``len`` is the token count, so this stub mirrors that
    shape exactly.
    """

    @staticmethod
    def encode(text: str, *, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens  # parity with HF signature
        return [0] * len(text.split())


def test_hf_hub_reuse_returns_none_for_a0_b0_c0_librarian() -> None:
    """The bug case: (A=0, B=0, C=0, D=1) librarian must not reuse the medium file.

    Pre-fix this returned ``leakage/marker_librarian_asst_excluded_medium.jsonl``
    if the file existed on HF Hub. Post-fix it returns ``None`` so the
    dispatch loop falls through to fresh Claude generation under the
    correct B=0 user suffix and band filter.
    """
    cell = Cell(a=0, b=0, c=0, d=1, e=0)
    assert _hf_hub_reuse_path("librarian", cell) is None


def test_hf_hub_reuse_returns_none_for_all_a0_b0_c0_sources() -> None:
    """Every source persona must skip the reuse path for the (A=0, B=0, C=0) recipe."""
    cell = Cell(a=0, b=0, c=0, d=1, e=0)
    for source in ("librarian", "surgeon", "programmer"):
        assert _hf_hub_reuse_path(source, cell) is None, (
            f"_hf_hub_reuse_path({source!r}, A=0/B=0/C=0) should be disabled; "
            "round-9 Fix E disables the HF Hub reuse short-circuit"
        )


def test_hf_hub_reuse_returns_none_for_other_cells() -> None:
    """Cells outside (A=0, B=0, C=0) were already ``None`` pre-fix; sanity-check.

    The pre-fix short-circuit only fired on the (A=0, B=0, C=0) tuple. After
    the fix every cell returns ``None``.
    """
    for a in (0, 1):
        for b in (0, 1):
            for c in (0, 1):
                cell = Cell(a=a, b=b, c=c, d=1, e=0)
                assert _hf_hub_reuse_path("librarian", cell) is None


# Round-9 revision-2 regression tests for the local-file cache B=0 guard.
#
# The local cache predicate at ``__main__._run_dispatch_mode`` calls
# ``_b0_cache_in_band_fraction`` to decide whether to reuse a B=0
# off-policy cache file. The dispatcher refuses (sets
# ``cache_acceptable=False``) when the returned fraction is below
# ``RELAXED_B1_UNDERFILL_FRACTION`` (0.5). These tests pin the helper.


def _row(role: str, n_words: int, *, stamped: bool = False) -> dict:
    """Build a fake pool row with a known whitespace-tokenizable completion."""
    row: dict = {"role": role, "completion": " ".join(["w"] * n_words)}
    if stamped:
        row["qwen_completion_tokens"] = n_words
    return row


def test_b0_cache_in_band_fraction_stale_medium_file_rejected() -> None:
    """Stale round-3 ``a0_b0_c0_offpolicy.jsonl`` with 231-480 token completions.

    The bug case: a local pool whose source-role completions match the
    leaked HF-Hub "medium" file's distribution (well above the 40-80
    token band). Predicate must return a fraction below 0.5 so the
    dispatcher refuses reuse.
    """
    rows = [_row("source", n) for n in (231, 250, 310, 400, 480)]
    in_band, n_source, frac = _b0_cache_in_band_fraction(rows, _StubTokenizer(), B_LENGTH_BANDS[0])
    assert n_source == 5
    assert in_band == 0
    assert frac == 0.0
    assert frac < RELAXED_B1_UNDERFILL_FRACTION


def test_b0_cache_in_band_fraction_clean_b0_pool_accepted() -> None:
    """Healthy in-band B=0 pool: predicate clears the 0.5 acceptance bar."""
    band_lo, band_hi = B_LENGTH_BANDS[0]
    rows = [_row("source", n) for n in (band_lo, band_lo + 5, (band_lo + band_hi) // 2, band_hi)]
    in_band, n_source, frac = _b0_cache_in_band_fraction(rows, _StubTokenizer(), B_LENGTH_BANDS[0])
    assert n_source == 4
    assert in_band == 4
    assert frac == 1.0
    assert frac >= RELAXED_B1_UNDERFILL_FRACTION


def test_b0_cache_in_band_fraction_no_source_rows_returns_zero() -> None:
    """Edge case: cache has bystander rows only — predicate returns 0.0 so dispatcher refuses."""
    rows = [_row("bystander", 60), _row("bystander", 60)]
    in_band, n_source, frac = _b0_cache_in_band_fraction(rows, _StubTokenizer(), B_LENGTH_BANDS[0])
    assert n_source == 0
    assert in_band == 0
    assert frac == 0.0
    assert frac < RELAXED_B1_UNDERFILL_FRACTION


def test_b0_cache_in_band_fraction_uses_stamped_token_count_when_present() -> None:
    """When ``qwen_completion_tokens`` is stamped, helper uses it instead of re-tokenizing.

    Synthesise a row whose stamped count says "in band" while the
    whitespace text is out of band. The helper must trust the stamp; if
    it didn't, the in_band fraction would drop to 0.
    """
    band_lo, band_hi = B_LENGTH_BANDS[0]
    in_band_count = (band_lo + band_hi) // 2
    # Whitespace text length = 5 words, well below the band, but the
    # stamp says we already verified the count upstream.
    rows = [
        {
            "role": "source",
            "completion": "a b c d e",
            "qwen_completion_tokens": in_band_count,
        }
    ]
    in_band, n_source, frac = _b0_cache_in_band_fraction(rows, _StubTokenizer(), B_LENGTH_BANDS[0])
    assert n_source == 1
    assert in_band == 1
    assert frac == 1.0


def test_b0_cache_in_band_fraction_below_half_refused() -> None:
    """Cache with exactly < 50% in-band source rows must be refused.

    Round-9 dispatcher: ``cache_acceptable = False`` when
    ``in_band_frac < RELAXED_B1_UNDERFILL_FRACTION`` (strict less-than).
    Five rows, two in band -> 0.4 < 0.5 -> refuse.
    """
    band_lo, band_hi = B_LENGTH_BANDS[0]
    rows = [
        _row("source", band_lo + 5),
        _row("source", band_hi - 5),
        _row("source", 300),
        _row("source", 350),
        _row("source", 400),
    ]
    in_band, n_source, frac = _b0_cache_in_band_fraction(rows, _StubTokenizer(), B_LENGTH_BANDS[0])
    assert n_source == 5
    assert in_band == 2
    assert frac == 0.4
    assert frac < RELAXED_B1_UNDERFILL_FRACTION
