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
import json
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


# ── §7 P0 verdict: per-arm grids + octave-shift re-pilot plan (unit 5) ─────────


def _verdict_args(root: Path, baseline: Path, grid_arm: list[str] | None = None):
    argv = ["--phase", "p0-verdict", "--eval-root", str(root), "--i778-baseline", str(baseline)]
    for spec in grid_arm or []:
        argv += ["--p0-grid-arm", spec]
    return J.build_argparser().parse_args(argv)


def _write_pilot_block(root: Path, sub: str, tag: str, model_mean: float) -> None:
    p = root / "pilot" / sub / "partial" / f"{tag}__evil.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"model_mean": model_mean}), encoding="utf-8")


def _seed_arm(root: Path, cfg: str, coefs, trait_means, coh_means) -> None:
    for coef, tm, cm in zip(coefs, trait_means, coh_means, strict=True):
        tag = f"{cfg}__evil__c{coef}"
        _write_pilot_block(root, "trait_scores", tag, tm)
        _write_pilot_block(root, "coherence", tag, cm)


def _baseline_file(tmp_path: Path, score: float = 50.0) -> Path:
    bl = tmp_path / "i778_baseline.json"
    bl.write_text(json.dumps({"trait_score": score}), encoding="utf-8")
    return bl


def test_p0_grids_default_and_per_arm_override():
    args = _verdict_args(Path("."), Path("."), ["A=0.25,0.75,1.5,2.5"])
    grids = J._p0_grids(args)
    assert grids["A"] == [0.25, 0.75, 1.5, 2.5]
    assert grids["C"] == [0.5, 1.5, 3.0, 5.0]  # unnamed arm keeps --p0-grid


def test_p0_grids_bad_spec_exits():
    import pytest

    args = _verdict_args(Path("."), Path("."), ["Z=1.0"])
    with pytest.raises(SystemExit, match="bad --p0-grid-arm"):
        J._p0_grids(args)


def test_p0_verdict_first_miss_emits_repilot_plan(tmp_path):
    """Arm A all-broken (coherence < 80 everywhere) -> octave x0.5 + a repilot
    block whose scaled cells match issue2225_train.synth_cell slugs exactly."""
    root = tmp_path / "eval_root"
    grid = [0.5, 1.5, 3.0, 5.0]
    _seed_arm(root, "A", grid, [30.0] * 4, [60.0] * 4)  # broken: coh < 80
    _seed_arm(root, "C", grid, [30.0] * 4, [90.0] * 4)  # brackets: coherent + suppressing
    rc = J.run_p0_verdict(_verdict_args(root, _baseline_file(tmp_path)))
    assert rc == J.RC_GATE_FAIL
    verdict = json.loads((root / "pilot_gate" / "p0_verdict.json").read_text())
    assert verdict["passed"] is False
    assert verdict["octave_shift"] == {"A": 0.5, "C": None}
    plan = verdict["repilot"]
    assert set(plan) == {"A"}
    assert plan["A"]["coef_scale"] == 0.5
    assert plan["A"]["grid_csv"] == "0.25,0.75,1.5,2.5"
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location("issue2225_train", _SCRIPTS / "issue2225_train.py")
    train = _ilu.module_from_spec(spec)
    sys.modules["issue2225_train"] = train
    spec.loader.exec_module(train)
    expected = [train.synth_cell("A", "evil", c).slug for c in (0.25, 0.75, 1.5, 2.5)]
    assert plan["A"]["cells"] == expected
    assert "--coef-scale 0.5" in plan["A"]["train_args"]


def test_p0_verdict_all_ineffective_recommends_x2(tmp_path):
    root = tmp_path / "eval_root"
    grid = [0.5, 1.5, 3.0, 5.0]
    # coherent everywhere but NEVER suppressing (trait_mean >= baseline)
    _seed_arm(root, "A", grid, [55.0] * 4, [90.0] * 4)
    _seed_arm(root, "C", grid, [30.0] * 4, [90.0] * 4)
    rc = J.run_p0_verdict(_verdict_args(root, _baseline_file(tmp_path)))
    assert rc == J.RC_GATE_FAIL
    verdict = json.loads((root / "pilot_gate" / "p0_verdict.json").read_text())
    assert verdict["octave_shift"]["A"] == 2.0
    assert verdict["repilot"]["A"]["grid_csv"] == "1.0,3.0,6.0,10.0"


def test_p0_verdict_passes_on_shifted_grid_with_positional_sign_coef(tmp_path):
    """Re-verdict under --p0-grid-arm: criterion (iii) anchors at the SAME grid
    position (second-largest coefficient) instead of the absolute 3.0."""
    root = tmp_path / "eval_root"
    shifted = [0.25, 0.75, 1.5, 2.5]
    _seed_arm(root, "A", shifted, [30.0] * 4, [90.0] * 4)
    _seed_arm(root, "C", [0.5, 1.5, 3.0, 5.0], [30.0] * 4, [90.0] * 4)
    args = _verdict_args(root, _baseline_file(tmp_path), ["A=0.25,0.75,1.5,2.5"])
    rc = J.run_p0_verdict(args)
    assert rc == 0
    verdict = json.loads((root / "pilot_gate" / "p0_verdict.json").read_text())
    assert verdict["passed"] is True
    crit = verdict["criteria"]["iii_A_sign_check_suppresses"]
    assert crit["sign_check_coef"] == 1.5  # second-largest of the shifted grid
    assert crit["passed"] is True
    assert verdict["repilot"] == {}  # nothing to shift on a pass


def test_p0_verdict_sign_failure_has_empty_repilot(tmp_path):
    """Criterion-(iii) failure while BOTH arms bracket -> no shift recommendation
    (the dispatcher's designed-halt rc=7 branch keys on the empty repilot)."""
    root = tmp_path / "eval_root"
    grid = [0.5, 1.5, 3.0, 5.0]
    # brackets (some coherent + suppressing coefs exist) but A@3.0 NOT suppressing
    _seed_arm(root, "A", grid, [30.0, 30.0, 55.0, 30.0], [90.0] * 4)
    _seed_arm(root, "C", grid, [30.0] * 4, [90.0] * 4)
    rc = J.run_p0_verdict(_verdict_args(root, _baseline_file(tmp_path)))
    assert rc == J.RC_GATE_FAIL
    verdict = json.loads((root / "pilot_gate" / "p0_verdict.json").read_text())
    assert verdict["passed"] is False
    assert verdict["octave_shift"] == {"A": None, "C": None}
    assert verdict["repilot"] == {}


# ── r2 blocker 3: sync-reissue resume-idempotency (g4 Major 3) ────────────────


def _reissue_fixture(tmp_path: Path):
    """One trait_scores unit: rollout 0 fully api-refusal-censored (2 draws),
    rollout 1 clean (2 kept draws — the parity candidate)."""
    tag, trait = "A__evil__c3.0", "evil"
    eval_root = tmp_path / "eval"
    pdir = eval_root / "trait_scores" / "partial"
    pdir.mkdir(parents=True)
    rollouts_dir = tmp_path / "rollouts"
    rollouts_dir.mkdir()
    (rollouts_dir / f"{tag}__{trait}.json").write_text(
        json.dumps(
            {
                "rows": [{"question": "q0", "rollouts": ["resp-0", "resp-1"]}],
                "n_questions": 1,
                "n_rollouts": 2,
            }
        )
    )
    block = {
        "tag": tag,
        "trait": trait,
        "rubric_id": trait,
        "arm": "A_evil_3.0",
        "n_questions": 1,
        "n_rollouts": 2,
        "per_question": [
            {
                "question_idx": 0,
                "rollout_scores": [None, 25.0],
                "rollout_draw_scores": [[], [20.0, 30.0]],
                "rollout_n_api_refusal": [2, 0],
                "rollout_n_transport_lost": [0, 0],
                "rollout_n_content_dropped": [0, 0],
                "mean": 25.0,
            }
        ],
        "model_mean": 25.0,
        "rate_gt50": 0.0,
        "n_rollouts_scored": 1,
        "n_rollouts_total": 2,
        "accounting": {
            "n_total_draws": 4,
            "n_content_dropped": 0,
            "n_refusal_draws": 0,
            "n_truncation_dropped": 0,
            "n_transport_lost": 0,
            "n_api_refusal": 2,
        },
        "judge_meta": {
            "judge_model": J.JUDGE_MODEL,
            "n_draws": 2,
            "temperature": J.JUDGE_TEMPERATURE,
            "max_tokens": J.JUDGE_MAX_TOKENS,
            "transport_mode": "batch",
        },
    }
    ppath = pdir / f"{tag}__{trait}.json"
    ppath.write_text(json.dumps(block))
    from types import SimpleNamespace

    args = SimpleNamespace(
        eval_root=str(eval_root),
        stage="final",
        external_root=str(tmp_path / "external"),
        cache_root=str(tmp_path / "cache"),
        save_raw_root=str(tmp_path / "raw"),
        rollouts_dir=str(rollouts_dir),
        narrow_rollouts_dir=str(tmp_path / "narrow"),
        parity_n=250,
    )
    return args, ppath


def test_sync_reissue_is_resume_idempotent(tmp_path, monkeypatch):
    """FAILS PRE-FIX (r2 blocker 3): a second run must NOT re-append the same
    cached sync draws into an already-merged unit (rollout_n_api_refusal never
    resets, so re-selection without the judge_meta guard double-merges)."""
    from explore_persona_space.eval import graded_judge as gj

    args, ppath = _reissue_fixture(tmp_path)
    calls: list[dict] = []

    def fake_rubric_for(rubric_id: str, external_root: Path):  # loader boundary
        return "rubric {question} {answer}", J.TRAIT_N_DRAWS

    def fake_judge_graded(  # signature mirrors eval.graded_judge.judge_graded
        items,
        eval_prompt,
        *,
        n_draws,
        cache_dir,
        save_raw,
        judge_model=gj.DEFAULT_JUDGE_MODEL,
        temperature=gj.DEFAULT_JUDGE_TEMPERATURE,
        max_tokens=64,
        dry_run=False,
        threshold_base=None,
    ):
        calls.append({"ids": [i[0] for i in items], "n_draws": n_draws})
        return gj.JudgeResult(
            scores={iid: 42.0 for iid, _q, _a in items},
            n_total_draws=n_draws * len(items),
            n_dropped_draws=0,
            per_item_scores={iid: [42.0] * n_draws for iid, _q, _a in items},
        )

    monkeypatch.setattr(J, "rubric_for", fake_rubric_for)
    monkeypatch.setattr(gj, "judge_graded", fake_judge_graded)

    J.run_sync_reissue(args)
    block1 = json.loads(ppath.read_text())
    q = block1["per_question"][0]
    assert q["rollout_draw_scores"][0] == [42.0, 42.0]  # merged sync draws
    assert q["rollout_draw_scores"][1] == [20.0, 30.0]  # untouched clean rollout
    assert q["rollout_scores"][0] == 42.0
    assert block1["judge_meta"]["api_refusal_reissue"]["n_draws_recovered"] == 2
    n_calls_run1 = len(calls)
    assert n_calls_run1 >= 1

    # Run 2 (the crash-partway resume): the unit must SKIP — draw lists stable,
    # zero further judge calls. Pre-fix this doubled the draw multiplicity.
    J.run_sync_reissue(args)
    block2 = json.loads(ppath.read_text())
    assert block2["per_question"][0]["rollout_draw_scores"] == q["rollout_draw_scores"]
    assert len(block2["per_question"][0]["rollout_draw_scores"][0]) == 2, "draws doubled on re-run"
    assert len(calls) == n_calls_run1, "re-run must dispatch no further judge calls"


def test_digest_reports_reissue_remediation(tmp_path, monkeypatch):
    """g4 minor: a completed sync-reissue is legible at the digest surface —
    per-row flag + remediation block + no stale 'run sync-reissue' warning."""
    from types import SimpleNamespace

    args, ppath = _reissue_fixture(tmp_path)
    block = json.loads(ppath.read_text())
    block["judge_meta"]["api_refusal_reissue"] = {
        "date": "2026-08-10",
        "n_draws_reissued": 2,
        "n_draws_recovered": 2,
        "path": "sync",
    }
    ppath.write_text(json.dumps(block))
    (tmp_path / "raw").mkdir(exist_ok=True)
    dargs = SimpleNamespace(
        eval_root=args.eval_root, stage="final", save_raw_root=str(tmp_path / "raw")
    )
    out = J.run_digest(dargs)
    digest = json.loads(Path(out).read_text())
    assert digest["api_refusal_remediation"] == {
        "n_units_reissued": 1,
        "n_draws_recovered": 2,
        "n_censored_units_unremediated": 0,
    }
    (row,) = digest["per_arm"]
    assert row["api_refusal_reissued"] is True
    assert row["n_draws_recovered_by_reissue"] == 2


# ── g3 Major 1: MMLU --limit threading (argv + resume fingerprint) ────────────


def test_mmlu_limit_threaded_into_argv_and_fingerprint(tmp_path):
    MM = _load("issue2225_mmlu")
    cmd = MM._lm_eval_cmd("pretrained=x", tmp_path, 200)
    assert cmd[cmd.index("--limit") + 1] == "200"
    assert "--limit" not in MM._lm_eval_cmd("pretrained=x", tmp_path, None)
    target = MM.evalgen.targets_by_tag()["base"]
    fp_probe = MM.mmlu_fingerprint(target, None, "m", limit=200)
    fp_full = MM.mmlu_fingerprint(target, None, "m", limit=None)
    assert fp_probe["limit"] == 200 and fp_full["limit"] is None
    # the P0 --limit probe must NEVER resume-satisfy P2c's full-set run
    assert fp_probe != fp_full
