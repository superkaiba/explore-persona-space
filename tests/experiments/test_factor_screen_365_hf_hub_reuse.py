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

Fix E disables the short-circuit unconditionally. These tests pin the
behaviour so a future "let's try the cache again" patch trips the test
suite immediately.
"""

from __future__ import annotations

from explore_persona_space.experiments.factor_screen_365.__main__ import (
    Cell,
    _hf_hub_reuse_path,
)


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
