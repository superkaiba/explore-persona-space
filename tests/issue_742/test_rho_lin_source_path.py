"""Test 5 (plan v7 §13 item 5 + §8 row + §12 row 4 + §6 measurement-validity) —
the loader reads #658 ridge rho_lin from analyzer_body_data.json at
/<genre>/a33/<beh>/lin_rho, NEVER from assumption_verdicts.json (which holds the
A3.2 MLP best_rho, a DIFFERENT quantity that would silently substitute).

Regression guard: a misread from assumption_verdicts.json would put broad_em at
~0.1458 (the A3.2 best_rho) vs the correct a33 lin_rho 0.444, and sycophancy at
~0.2151 vs the correct 0.1268 — the test catches that source confusion.

On-disk values verified this session against the worktree's committed
eval_results/issue_658/ (sparse cone present):
  betley/a33 lin_rho:  broad_em 0.444, harmful_compliance 0.6921,
                       sycophancy 0.1268, refusal 0.4223
  g1/a33 lin_rho:      broad_em 0.3981, harmful_compliance 0.6112,
                       sycophancy 0.2486, refusal 0.5698
  assumption_verdicts a32 best_rho (WRONG source): broad_em 0.14584..., etc.
"""

from __future__ import annotations

import json

import pytest

from explore_persona_space.task_workflow import repo_root

from .conftest import impl, impl_has

EVAL_DIR = repo_root() / "eval_results" / "issue_658"

# The bracket interpretation depends on these on-disk Betley a33 lin_rho values.
BETLEY_EXPECTED = {
    "broad_em": 0.444,
    "harmful_compliance": 0.6921,
    "sycophancy": 0.1268,
    "refusal": 0.4223,
}
G1_EXPECTED = {
    "broad_em": 0.3981,
    "harmful_compliance": 0.6112,
    "sycophancy": 0.2486,
    "refusal": 0.5698,
}
TOL = 1e-4


# --------------------------------------------------------------------------- #
# The on-disk artifacts carry the documented shape (no impl needed; data-truth) #
# --------------------------------------------------------------------------- #
def test_analyzer_body_data_carries_a33_lin_rho_betley():
    abd = json.loads((EVAL_DIR / "analyzer_body_data.json").read_text())
    a33 = abd["betley"]["a33"]
    for beh, expected in BETLEY_EXPECTED.items():
        got = a33[beh]["lin_rho"]
        assert abs(got - expected) <= TOL, (
            f"betley/a33/{beh}/lin_rho on disk = {got} != expected {expected}"
        )


def test_analyzer_body_data_carries_a33_lin_rho_g1():
    abd = json.loads((EVAL_DIR / "analyzer_body_data.json").read_text())
    a33 = abd["g1"]["a33"]
    for beh, expected in G1_EXPECTED.items():
        got = a33[beh]["lin_rho"]
        assert abs(got - expected) <= TOL, (
            f"g1/a33/{beh}/lin_rho on disk = {got} != expected {expected}"
        )


def test_assumption_verdicts_holds_a_DIFFERENT_quantity():
    # the regression-guard premise: assumption_verdicts.json carries the A3.2
    # best_rho (broad_em ~= 0.1458), which is NOT the a33 ridge lin_rho (0.444).
    av = json.loads((EVAL_DIR / "assumption_verdicts.json").read_text())
    best_rho_broad_em = av["a32_verdicts"]["broad_em"]["best_rho"]
    assert abs(best_rho_broad_em - 0.444) > 0.2, (
        "regression-guard premise broken: assumption_verdicts best_rho should be "
        f"a DIFFERENT quantity from the a33 lin_rho, got {best_rho_broad_em}"
    )
    # and it differs materially from EVERY read-out behavior's a33 lin_rho
    assert abs(best_rho_broad_em - BETLEY_EXPECTED["broad_em"]) > 0.2


# --------------------------------------------------------------------------- #
# The loader reads the RIGHT source + value (impl-guarded)                      #
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(
    not impl_has("load_rho_lin"),
    reason="implementation pending round 2",
)
@pytest.mark.parametrize("beh,expected", sorted(BETLEY_EXPECTED.items()))
def test_load_rho_lin_betley_reads_a33_value(beh, expected):
    got = impl.load_rho_lin(beh, "betley", eval_dir=EVAL_DIR)
    assert abs(got - expected) <= TOL, (
        f"load_rho_lin('{beh}','betley') = {got} != on-disk a33 lin_rho {expected}"
    )


@pytest.mark.skipif(
    not impl_has("load_rho_lin"),
    reason="implementation pending round 2",
)
@pytest.mark.parametrize("beh,expected", sorted(G1_EXPECTED.items()))
def test_load_rho_lin_g1_reads_a33_value(beh, expected):
    got = impl.load_rho_lin(beh, "ultrachat", eval_dir=EVAL_DIR)
    assert abs(got - expected) <= TOL, (
        f"load_rho_lin('{beh}','ultrachat') = {got} != on-disk g1 a33 lin_rho {expected}"
    )


@pytest.mark.skipif(
    not impl_has("load_rho_lin"),
    reason="implementation pending round 2",
)
def test_load_rho_lin_never_silently_substitutes_best_rho():
    # the load_rho_lin loader must NOT return the A3.2 best_rho. broad_em is the
    # sharpest tell: a33 lin_rho=0.444 vs assumption_verdicts best_rho=0.1458.
    got = impl.load_rho_lin("broad_em", "betley", eval_dir=EVAL_DIR)
    av = json.loads((EVAL_DIR / "assumption_verdicts.json").read_text())
    wrong = av["a32_verdicts"]["broad_em"]["best_rho"]
    assert abs(got - wrong) > 0.2, (
        f"load_rho_lin returned {got}, dangerously close to the WRONG best_rho "
        f"{wrong} from assumption_verdicts.json"
    )
    assert abs(got - BETLEY_EXPECTED["broad_em"]) <= TOL


@pytest.mark.skipif(
    not impl_has("load_rho_lin"),
    reason="implementation pending round 2",
)
def test_load_rho_lin_raises_when_pointed_at_assumption_verdicts(tmp_path):
    # if the loader is (mis)pointed at a dir whose a33 source is actually the
    # assumption_verdicts object, it must RAISE a clear error rather than silently
    # parse best_rho as if it were lin_rho (fail-loud, CLAUDE.md Critical Rules).
    bad_dir = tmp_path / "bad_eval"
    bad_dir.mkdir()
    # write a decoy analyzer_body_data.json that lacks the /a33/<beh>/lin_rho key
    # (simulating a source that does not carry the ridge value)
    (bad_dir / "analyzer_body_data.json").write_text(
        json.dumps({"betley": {"a32": {"broad_em": {"best_rho": 0.1458}}}})
    )
    with pytest.raises(Exception):  # noqa: B017 - any clear error; never a silent substitute
        impl.load_rho_lin("broad_em", "betley", eval_dir=bad_dir)
