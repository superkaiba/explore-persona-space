"""Issue #928 invariants: group-fold batched ridge parity + CoT segmentation.

Pins two permanent invariants of the #928 CoT-decomposition pipeline:

1. The batched GROUP-fold LOCO/LOFO ridge (``issue928_null_bootstrap``) must
   reproduce the serial references — ``ridge_predict_loco_centered`` (the
   committed #722/#810 estimator) on singleton groups, and an inline serial
   oracle on multi-row groups + null draws — at atol 1e-8 (vectorize-rule
   item 6). A refactor that silently changes the PRESS/dual identities or the
   group-fold train-mean baseline fails here before it can ship wrong skills.
2. The rung-aware ``segment_completion`` parser (plan §4.4) — including the
   rung-(iii) prefill criterion adjustment and the malformed-reason taxonomy —
   and the BPE-merge-robust ``char_span_to_token_span`` overlap semantics
   (the #825 zero-width-span guard returns (0, 0), never a crash).
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))


def test_group_ridge_matches_serial_references():
    from issue928_null_bootstrap import assert_group_ridge_matches_serial

    devs = assert_group_ridge_matches_serial(seed=928, atol=1e-8)
    assert devs, "parity gate returned no checks"
    assert max(devs.values()) < 1e-8


def test_segment_completion_rungs_and_reasons():
    from issue928_common import segment_completion

    ok, reason, cot, ans = segment_completion(
        "<think>\nreasoning here\n</think>\n\nfinal answer", "greedy"
    )
    assert ok and reason == ""
    assert cot == (len("<think>"), len("<think>") + len("\nreasoning here\n"))
    assert ans[1] > ans[0]

    # rung (iii): no <think> requirement; CoT = start .. before </think>.
    ok, reason, cot, _ans = segment_completion("prefilled thoughts\n</think>\n\nanswer", "prefill")
    assert ok and cot[0] == 0

    for text, rung, want_reason in [
        ("no block at all", "greedy", "no_close"),
        ("<think>\nr\n</think> x </think> y", "greedy", "multiple_close"),
        ("pre <think>\nr\n</think>\nans", "greedy", "think_not_at_start"),
        ("<think>\n\n</think>\n\nans", "greedy", "empty_cot"),
        ("<think>\nr\n</think>\n\n  ", "greedy", "empty_answer"),
        ("r\n</think>\n\nans" + "<think>", "prefill", ""),  # prefill ignores <think>
    ]:
        ok, reason, _c, _a = segment_completion(text, rung)
        assert reason == want_reason, (text, reason)


def test_char_span_to_token_span_overlap_and_zero_width():
    from issue928_common import char_span_to_token_span

    offsets = [(0, 3), (3, 5), (5, 9), (9, 12)]
    assert char_span_to_token_span(offsets, (3, 9)) == (1, 3)
    # partial overlap includes the straddling token (BPE-merge robustness).
    assert char_span_to_token_span(offsets, (4, 6)) == (1, 3)
    # zero-width / out-of-range span -> (0, 0) sentinel (caller drops the row).
    assert char_span_to_token_span(offsets, (12, 12)) == (0, 0)


def test_group_perm_matrix_preserves_group_blocks():
    import numpy as np
    from issue928_null_bootstrap import make_group_perm_matrix

    groups = np.repeat(np.arange(4), [3, 2, 3, 2])
    perm = make_group_perm_matrix(groups, [0, 1, 2, 3], 8, np.random.default_rng(0))
    assert perm.shape == (8, 10)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in range(4)}
    for b in range(8):
        for g in range(4):
            src_groups = {int(groups[i]) for i in perm[b][rows_by_group[g]]}
            assert len(src_groups) == 1  # a whole block maps to ONE source group
