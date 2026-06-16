"""Task #653 §3.4.CI deciding-DV ambiguity flag (round-4 BLOCKER deciding-ci-hardcoded-top-share).

CPU-only. The per-cell ambiguity flag MUST be bootstrapped on the DV that
DECIDED the cell's H-label, and checked against THAT DV's own threshold(s) —
never a hardcoded top-share whose [0,1] CI can never bracket the PR thresholds
2.0/5.0. ``classify_cell`` returns ``deciding_dv`` and selects the bootstrap DV
via :func:`deciding_dv_for_label`; the on-pod analyze + the off-pod refresh both
use this identical logic (no module-level ``DECIDING_DV = "top_share_lambda"``).
"""

from __future__ import annotations

import re
from pathlib import Path

from explore_persona_space.experiments.issue_653 import spectral

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _classify(spec, *, cos=None, random_ci_high=None, ci=None):
    """Run classify_cell with a stubbed bootstrap returning a fixed CI for the
    deciding DV (so the test controls the CI without a real cloud)."""
    return spectral.classify_cell(
        cell_group="g",
        rung="r16",
        spec=spec,
        n_rows=20,
        cos_top_to_rb=cos,
        random_ci_high=random_ci_high,
        bootstrap_fn=(lambda _dv, _ci=ci: _ci) if ci is not None else None,
    )


def test_pr_decided_h3_uses_pr_ci():
    """top_share=0.55, pr=6.0, rank_k=4 → H3 via PR only → deciding_dv == 'pr_lambda'.
    A PR CI that BRACKETS the PR threshold 5.0 → ambiguous True; a PR CI entirely
    above 5.0 → ambiguous False. FAILS on round-4 code (always bootstraps top-share,
    whose [0,1] CI never reaches 5.0 → always ambiguous=False).

    NB: the v5 §3.4.CI spec wrote the ambiguous CI as (5.2, 6.8), but per the
    BINDING pseudocode ``ambiguous = any(ci_lo <= thr <= ci_hi)`` a CI must CONTAIN
    the threshold to flag — and 5.2 > 5.0, so (5.2, 6.8) is actually entirely above
    5.0 (a spec arithmetic slip; the spec's own (5.6, 6.8)=non-ambiguous case
    confirms a fully-above CI is unambiguous). We use a CI that genuinely straddles
    5.0 for the ambiguous case, matching the pseudocode contract exactly."""
    spec = {"top_share_lambda": 0.55, "pr_lambda": 6.0, "rank_k_at_90": 4}
    base = _classify(spec)
    assert base.label == "H3"
    assert base.deciding_dv == "pr_lambda"

    amb = _classify(spec, ci=(4.8, 6.8))  # straddles 5.0 → contains the threshold
    assert amb.deciding_dv == "pr_lambda"
    assert amb.ambiguous is True

    clear = _classify(spec, ci=(5.6, 6.8))  # entirely above 5.0 → unambiguous
    assert clear.deciding_dv == "pr_lambda"
    assert clear.ambiguous is False


def test_rank_k_decided_h3_uses_rank_k_ci():
    """top_share=0.60, pr=3.0, rank_k=11 → H3 via rank-K only → deciding_dv ==
    'rank_k_at_90'; CI (9,13) brackets 10 → ambiguous True; (11,14) → False."""
    spec = {"top_share_lambda": 0.60, "pr_lambda": 3.0, "rank_k_at_90": 11}
    base = _classify(spec)
    assert base.label == "H3"
    assert base.deciding_dv == "rank_k_at_90"

    amb = _classify(spec, ci=(9.0, 13.0))  # brackets 10
    assert amb.ambiguous is True

    clear = _classify(spec, ci=(11.0, 14.0))  # at/above 10
    assert clear.ambiguous is False


def test_top_share_decided_h1_uses_top_share_ci():
    """top_share=0.72, pr=3.0, rank_k=4, alignment-not-read (cos=None) →
    deciding_dv == 'top_share_lambda' (the only low-rank criterion met); CI
    (0.66, 0.78) brackets 0.7 → ambiguous True."""
    spec = {"top_share_lambda": 0.72, "pr_lambda": 3.0, "rank_k_at_90": 4}
    base = _classify(spec)
    assert base.deciding_dv == "top_share_lambda"

    amb = _classify(spec, ci=(0.66, 0.78))  # brackets 0.7
    assert amb.deciding_dv == "top_share_lambda"
    assert amb.ambiguous is True

    clear = _classify(spec, ci=(0.72, 0.80))  # above 0.7
    assert clear.ambiguous is False


def test_cos_decided_h1_marks_ci_unavailable():
    """An aligned low-rank cell (cos read present) is H1↔H2 decided by the cosine
    + #503 random-CI exceedance — a cluster bootstrap is not meaningful. deciding_dv
    == 'cos_top_to_rb', deciding_ci_unavailable True with the explicit reason — NOT
    a silent top-share fallback (§3.4.CI cos branch)."""
    spec = {"top_share_lambda": 0.72, "pr_lambda": 3.0, "rank_k_at_90": 4}
    v = _classify(spec, cos=0.8, random_ci_high=0.1)
    assert v.label == "H1"
    assert v.deciding_dv == "cos_top_to_rb"
    assert v.deciding_ci_unavailable is True
    assert "alignment-driven" in v.deciding_ci_reason
    # the alignment-ambiguity fires when |cos| ≤ the random-CI upper bound:
    amb = _classify(spec, cos=0.55, random_ci_high=0.6)
    assert amb.deciding_dv == "cos_top_to_rb"
    assert amb.ambiguous is True


def test_boundary_label_has_no_deciding_dv_and_is_ambiguous():
    """A neither-low-rank-nor-H3 boundary cell has no single deciding DV →
    deciding_dv None, deciding_ci_unavailable True, ambiguous True (§3.4.CI)."""
    spec = {"top_share_lambda": 0.5, "pr_lambda": 3.0, "rank_k_at_90": 4}
    v = _classify(spec)
    assert v.deciding_dv is None
    assert v.deciding_ci_unavailable is True
    assert v.ambiguous is True


def test_no_module_level_hardcoded_deciding_dv():
    """Static check: neither i653_postpod_bootstrap.py nor i653_dispatch.py
    contains a module-level ``DECIDING_DV = "top_share_lambda"`` or a literal
    ``cluster_bootstrap_dv(..., "top_share_lambda", ...)`` — the deciding DV is
    always selected from the label, never hardcoded (§3.4.CI rule 1)."""
    for rel in (
        "scripts/issue_653/i653_postpod_bootstrap.py",
        "scripts/issue_653/i653_dispatch.py",
    ):
        text = (_REPO_ROOT / rel).read_text()
        assert 'DECIDING_DV = "top_share_lambda"' not in text, rel
        # no literal-DV-name call to cluster_bootstrap_dv (the round-4 bug shape):
        hits = re.findall(r'cluster_bootstrap_dv\([^)]*"top_share_lambda"', text, re.DOTALL)
        assert not hits, f"{rel}: hardcoded top-share bootstrap call {hits}"


def test_deciding_dv_for_label_precedence():
    """deciding_dv_for_label: H3 wins first; tightest-margin H3 criterion when both
    PR and rank-K cross; low-rank top-share/PR otherwise; None at the boundary."""
    f = spectral.deciding_dv_for_label
    # both H3 criteria cross — pick the tightest fractional margin:
    #   pr=5.5 (margin 0.10) vs rank_k=20 (margin 1.0) → pr_lambda wins
    assert (
        f(
            top_share=0.5,
            pr=5.5,
            rank_k=20.0,
            cos_top_to_rb=None,
            is_low_rank=False,
            is_h3=True,
            is_aligned=None,
        )
        == "pr_lambda"
    )
    # boundary: neither low-rank nor H3 → None
    assert (
        f(
            top_share=0.5,
            pr=3.0,
            rank_k=4.0,
            cos_top_to_rb=None,
            is_low_rank=False,
            is_h3=False,
            is_aligned=None,
        )
        is None
    )
