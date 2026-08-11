"""Issue #2225 — unit-4 pins: judge instrument contract + analysis statistics.

Covers (plan §3/§6/§4.7, unit-4 verification contract):
  1. Rule-27 parse-contract round-trip for the NARROW mistake-style rubric —
     realistic reason-then-score replies (plain + fenced) through the REAL
     ``parse_judge_json`` / ``_score_from_parsed`` reduce, plus the
     {question}/{answer} substitution-placeholder presence checks.
  2. Arm identity mapping: ``parse_tag`` / ``arm_filename`` / ``safe_item_id``
     (batch custom-id charset + "__" ban + length budget).
  3. Synthetic bootstrap/selection fixtures: the §3 lattice verdict's
     exhaustive three-way partition, matched-coherence selection, the frozen
     paired bootstrap, and the SELECTION-INHERITED variant (constant-selection
     equivalence + no-coherent-coefficient invalid draws).
  4. Probe-fit math through the batched Gram-ridge path (vectorize-many-cell
     fits shape): exact kernel double-centering vs explicit centering, batched
     solve vs per-layer reference, orthogonal-complement projection, rank AUC.

CPU-only, no network, repo-root-relative paths (sparse-worktree safe: reads no
``eval_results/issue_<M>`` fixtures).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


J = _load("issue2225_judge")
A = _load("issue2225_analysis")


# ── 1. rule-27 parse-contract round-trip (narrow rubric) ──────────────────────


def _reduce(text: str) -> float | None:
    from explore_persona_space.eval.graded_judge import _score_from_parsed
    from explore_persona_space.eval.utils import parse_judge_json

    return _score_from_parsed(parse_judge_json(text))


def test_narrow_rubric_round_trips_realistic_reply():
    reply = (
        '{"reasoning": "The answer endorses the factually wrong opinion with '
        'confident framing, matching the trained mistake style.", "score": 85}'
    )
    assert _reduce(reply) == 85.0


def test_narrow_rubric_round_trips_fenced_reply():
    reply = (
        "Here is my assessment:\n```json\n"
        '{"reasoning": "Accurate, hedged answer; no mistake-style endorsement.",\n'
        ' "score": 10}\n```'
    )
    assert _reduce(reply) == 10.0


def test_narrow_rubric_refusal_and_out_of_range_drop():
    assert _reduce('{"reasoning": "cannot assess", "score": "REFUSAL"}') is None
    assert _reduce('{"reasoning": "x", "score": 250}') is None  # out-of-range -> drop


def test_narrow_rubric_has_substitution_placeholders():
    assert "{question}" in J.NARROW_RUBRIC and "{answer}" in J.NARROW_RUBRIC
    # harness-identical substitution (graded_judge.format_user_msg uses
    # str.replace) leaves no slot unfilled
    rendered = J.NARROW_RUBRIC.replace("{question}", "Q?").replace("{answer}", "A.")
    assert "{question}" not in rendered and "{answer}" not in rendered


def test_narrow_rubric_anchors_present():
    for anchor in ("0", "50", "100"):
        assert anchor in J.NARROW_RUBRIC  # anchored rubric (llm-judging rule 6)


# ── 2. arm identity mapping ───────────────────────────────────────────────────


def test_parse_tag_cell_baseline_base():
    cell = J.parse_tag("A__evil__c3.0")
    assert (cell.kind, cell.config, cell.dataset, cell.coef_tag) == ("cell", "A", "evil", "3.0")
    prompt = J.parse_tag("H__evil")
    assert (prompt.kind, prompt.config, prompt.coef_tag) == ("cell", "H", "prompt")
    baseline = J.parse_tag("baseft_mistake_opinions")
    assert (baseline.kind, baseline.dataset) == ("baseline", "mistake_opinions")
    assert J.parse_tag("base").kind == "base"
    with pytest.raises(ValueError):
        J.parse_tag("not_a_tag")


def test_arm_filename_matches_analysis_arm_path():
    assert J.arm_filename("A__evil__c3.0") == "A_evil_3.0.json"
    assert J.arm_filename("H__evil") == "H_evil_prompt.json"
    assert J.arm_filename("baseft_evil") == "baseline_evil.json"
    assert J.arm_filename("base") == "base.json"
    # the analysis reader composes the SAME name (cross-script contract)
    assert A._arm_path(Path("x"), "trait_scores", "A", "evil", 3.0).name == "A_evil_3.0.json"
    assert A._arm_path(Path("x"), "coherence", "H", "evil", None).name == "H_evil_prompt.json"


def test_safe_item_id_charset_and_budget():
    iid = J.safe_item_id("D__mistake_opinions__c0.75", 19, 9)
    assert "__" not in iid and len(iid) <= 40
    assert all(c.isalnum() or c in "_-" for c in iid)
    # bijective over the realized tag set: distinct tags stay distinct
    ids = {
        J.safe_item_id(tag, 0, 0)
        for tag in ("A__evil__c0.5", "A__evil__c5.0", "H__evil", "base", "baseft_evil")
    }
    assert len(ids) == 5


# ── 3. lattice + selection + bootstrap fixtures ───────────────────────────────


def test_lattice_verdict_exhaustive_partition():
    assert A.lattice_verdict(-2.0, -3.0, -1.0) == "Context-position-superior"
    assert A.lattice_verdict(2.0, 1.0, 3.0) == "Context-position-inferior"
    assert A.lattice_verdict(-0.5, -2.0, 1.0) == "Statistical tie"  # CI straddles 0
    assert A.lattice_verdict(0.0, -1.0, 1.0) == "Statistical tie"  # point exactly 0
    # exhaustive: every (point, lo, hi) shape lands in exactly one class
    rng = np.random.default_rng(0)
    for _ in range(200):
        pt = float(rng.normal())
        lo, hi = sorted(rng.normal(size=2))
        assert A.lattice_verdict(pt, float(lo), float(hi)) in {
            "Context-position-superior",
            "Context-position-inferior",
            "Statistical tie",
        }


def test_matched_coherence_select_largest_eligible():
    assert A.matched_coherence_select({0.5: 95.0, 1.5: 88.0, 3.0: 81.0, 5.0: 60.0}) == 3.0
    assert A.matched_coherence_select({0.5: 79.9, 1.5: 60.0}) is None
    assert A.matched_coherence_select({0.5: 80.0}) == 0.5  # threshold inclusive
    assert A.matched_coherence_select({0.5: None, 1.5: 90.0}) == 1.5  # None never eligible


def test_paired_bootstrap_ci_deterministic_and_covering():
    delta = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    p1 = A.paired_bootstrap_ci(delta, 500, seed=7)
    p2 = A.paired_bootstrap_ci(delta, 500, seed=7)
    assert p1[0] == 3.0 and p1[1] == p2[1] and p1[2] == p2[2]  # seeded reproducible
    assert p1[1] <= p1[0] <= p1[2]  # CI brackets the point estimate
    # a constant delta has a degenerate CI at the constant
    pc = A.paired_bootstrap_ci(np.full(8, 2.5), 200, seed=1)
    assert pc[1] == pytest.approx(2.5) and pc[2] == pytest.approx(2.5)


def _curve(coefs, trait_rows, coh_rows, steered_n=100.0, fixed_sum=0.0, fixed_n=0.0):
    n_coef = len(coefs)
    return A.ArmCurve(
        coefs=list(coefs),
        trait_qc=np.asarray(trait_rows, dtype=np.float64),
        coh_qc=np.asarray(coh_rows, dtype=np.float64),
        coh_steered_n=np.full(n_coef, steered_n),
        coh_fixed_sum=np.full(n_coef, fixed_sum),
        coh_fixed_n=np.full(n_coef, fixed_n),
    )


def test_selection_inherited_equals_frozen_under_constant_selection():
    # both coefs coherent in EVERY resample -> selection constant at the
    # largest coef -> inherited draws == frozen draws at that coef.
    n_q, n_boot = 6, 300
    rng = np.random.default_rng(3)
    trait_x = rng.uniform(10, 30, size=(2, n_q))
    trait_y = rng.uniform(20, 40, size=(2, n_q))
    coh = np.full((2, n_q), 95.0)
    arm_x = _curve([1.0, 2.0], trait_x, coh)
    arm_y = _curve([1.0, 2.0], trait_y, coh)
    idx = np.random.default_rng(11).integers(0, n_q, size=(n_boot, n_q))
    delta, n_invalid = A.selection_inherited_delta_draws(arm_x, arm_y, idx)
    assert n_invalid == 0
    expect = (trait_x[1] - trait_y[1])[idx].mean(axis=1)  # frozen at coef 2.0
    np.testing.assert_allclose(delta, expect, rtol=1e-12)


def test_selection_inherited_flags_no_coherent_draws_invalid():
    # coherence straddles 80 per question -> some resamples leave NO eligible
    # coefficient for arm_x -> those draws are NaN-counted, never coerced.
    n_q, n_boot = 4, 400
    coh_x = np.array([[75.0, 75.0, 88.0, 88.0]])  # single coef; mean depends on draw
    trait = np.full((1, n_q), 50.0)
    arm_x = _curve([1.0], trait, coh_x)
    arm_y = _curve([1.0], trait, np.full((1, n_q), 95.0))
    idx = np.random.default_rng(5).integers(0, n_q, size=(n_boot, n_q))
    delta, n_invalid = A.selection_inherited_delta_draws(arm_x, arm_y, idx)
    assert 0 < n_invalid < n_boot
    assert int(np.isnan(delta).sum()) == n_invalid


def test_selection_inherited_blends_fixed_other_trait_coherence():
    # opinions-style arm: steered-trait coherence 90 but a large FIXED
    # other-trait pool at 40 drags the blended statistic below 80 -> no
    # eligible coefficient in ANY draw.
    n_q, n_boot = 4, 50
    trait = np.full((1, n_q), 50.0)
    coh = np.full((1, n_q), 90.0)
    arm_x = _curve([1.0], trait, coh, steered_n=100.0, fixed_sum=40.0 * 300.0, fixed_n=300.0)
    arm_y = _curve([1.0], trait, coh)  # fixed_n=0 -> pure steered coherence (eligible)
    idx = np.random.default_rng(9).integers(0, n_q, size=(n_boot, n_q))
    delta, n_invalid = A.selection_inherited_delta_draws(arm_x, arm_y, idx)
    assert n_invalid == n_boot  # blended 90*(100/400)+40*(300/400)=52.5 < 80 always
    assert np.isnan(delta).all()


# ── 4. probe math through the batched Gram-ridge path ─────────────────────────


def test_center_gram_matches_explicit_centering():
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    n, L, d = 30, 3, 7
    X = torch.randn(n, L, d, dtype=torch.float64)
    K = torch.einsum("nld,mld->lnm", X, X)
    tr = torch.arange(0, 20)
    te = torch.arange(20, n)
    K_trtr_c, K_tetr_c = A._center_gram(K, tr, te)
    for li in range(L):
        Xtr, Xte = X[tr, li, :], X[te, li, :]
        mu = Xtr.mean(dim=0)
        ref_trtr = (Xtr - mu) @ (Xtr - mu).T
        ref_tetr = (Xte - mu) @ (Xtr - mu).T
        torch.testing.assert_close(K_trtr_c[li], ref_trtr, rtol=1e-9, atol=1e-9)
        torch.testing.assert_close(K_tetr_c[li], ref_tetr, rtol=1e-9, atol=1e-9)


def test_batched_ridge_solve_matches_per_layer_reference():
    torch = pytest.importorskip("torch")
    torch.manual_seed(1)
    L, n = 4, 12
    X = torch.randn(L, n, 6, dtype=torch.float64)
    K_c = torch.einsum("lnd,lmd->lnm", X, X)
    y = torch.randn(n, dtype=torch.float64)
    lam = torch.tensor([0.1, 1.0, 5.0, 0.5], dtype=torch.float64)
    alpha = A._batched_ridge_solve(K_c, y, lam)
    assert alpha.shape == (L, n)
    for li in range(L):
        ref = torch.linalg.solve(K_c[li] + lam[li] * torch.eye(n, dtype=torch.float64), y)
        torch.testing.assert_close(alpha[li], ref, rtol=1e-8, atol=1e-8)


def test_batched_ridge_solve_singular_slice_falls_back_to_pinv():
    torch = pytest.importorskip("torch")
    n = 5
    healthy = torch.eye(n) * 2.0
    singular = torch.zeros(n, n)  # exactly singular slice
    K_c = torch.stack([healthy, singular])
    y = torch.ones(n)
    alpha = A._batched_ridge_solve(K_c, y, torch.tensor([0.0, 0.0]))
    assert torch.isfinite(alpha).all()  # pinv fallback, never a raise/placeholder
    torch.testing.assert_close(alpha[0], y / 2.0)


def test_project_out_removes_direction_component():
    torch = pytest.importorskip("torch")
    torch.manual_seed(2)
    n, L, d = 10, 3, 8
    X = torch.randn(n, L, d)
    v = torch.randn(L, d)
    Xp = A._project_out(X, v)
    vhat = v / v.norm(dim=1, keepdim=True)
    residual = torch.einsum("nld,ld->nl", Xp, vhat)
    assert residual.abs().max().item() < 1e-5
    # idempotent: projecting twice changes nothing
    torch.testing.assert_close(A._project_out(Xp, v), Xp, rtol=1e-5, atol=1e-6)


def test_auc_separable_and_chance():
    assert A._auc([1.0, 2.0, 3.0, 10.0, 11.0, 12.0], [-1, -1, -1, 1, 1, 1]) == 1.0
    assert A._auc([5.0, 5.0, 5.0, 5.0], [-1, -1, 1, 1]) == 0.5  # all-tied -> midrank 0.5
