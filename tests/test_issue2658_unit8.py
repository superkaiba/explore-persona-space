"""Issue #2658 unit-8 tests: discordance / clustered power / cost / gate verdict.

Statistical-content coverage (the unit-8 brief's mandated list):
- known-AUROC fixtures recover the right value (sklearn parity incl. ties, and
  binormal-shift calibration);
- single-class prompts are EXCLUDED from the equal-prompt macro, never imputed;
- the permutation stays WITHIN exact prompt (per-prompt label multisets
  preserved across every draw);
- the 10-vs-30-draw discordance projection is strictly larger for interior p
  (and the plug-in degenerates at p in {0,1} where the Jeffreys form does not);
- the power-unit ledger resumes (skips completed units; recomputes when any
  generating parameter changes);
- NOT-ESTIMABLE never becomes PASS (verdict PARKs);
- a projection is never reported as measured (basis labels + named missing
  artifacts).

All tests are OFFLINE and synthetic: no GPU, no judge API call, no bank item
text. Fixtures that mimic pilot artifacts use the PRODUCTION schemas
(i2658-judge-cell-v1 / i2658-objective-labels-v1) so the real loader bodies
execute.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2658_common as C  # noqa: E402
import issue2658_judge as J  # noqa: E402
import issue2658_power as P  # noqa: E402

JUDGED_ROW = "casualness"  # benign judged row
OBJECTIVE_ROW = "correctness_math"


# ---------------------------------------------------------------------------
# Deliverable A — estimator primitives.
# ---------------------------------------------------------------------------
def test_within_prompt_auroc_matches_sklearn_with_ties():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(120):
        n = int(rng.integers(4, 40))
        scores = rng.integers(0, 6, n).astype(float)  # heavy ties => midranks matter
        labels = rng.random(n) < 0.5
        if labels.all() or not labels.any():
            assert np.isnan(P.within_prompt_auroc(scores, labels))
            continue
        got = P.within_prompt_auroc(scores, labels)
        want = roc_auc_score(labels.astype(int), scores)
        assert abs(got - want) < 1e-12
        checked += 1
    assert checked > 60


def test_known_auroc_recovery_binormal():
    rng = np.random.default_rng(1)
    delta = P.binormal_shift(0.75)
    labels = rng.random(20_000) < 0.5
    scores = rng.standard_normal(20_000) + delta * labels
    assert abs(P.within_prompt_auroc(scores, labels) - 0.75) < 0.02


def test_single_class_prompt_excluded_from_macro():
    rng = np.random.default_rng(2)
    pid = np.array(["a"] * 6 + ["b"] * 5 + ["c"] * 4)
    scores = rng.normal(size=15)
    labels = rng.random(15) < 0.5
    labels[11:15] = True  # prompt c single-class
    labels[0], labels[1], labels[6], labels[7] = True, False, True, False  # a, b discordant
    macro, n_disc = P.equal_prompt_macro_auroc(scores, labels, pid)
    manual = []
    for g in ("a", "b", "c"):
        m = pid == g
        if 0 < labels[m].sum() < m.sum():
            manual.append(roc_auc_score(labels[m].astype(int), scores[m]))
    assert n_disc == len(manual) == 2
    assert abs(macro - float(np.mean(manual))) < 1e-12
    # every prompt single-class => NaN macro, zero discordant — never 0.5-imputed
    macro0, n0 = P.equal_prompt_macro_auroc(scores, np.ones(15, dtype=bool), pid)
    assert np.isnan(macro0) and n0 == 0


def test_flat_layout_matches_balanced_layout():
    rng = np.random.default_rng(3)
    n_prompts, n_resp = 7, 6
    scores = rng.normal(size=(n_prompts, n_resp))
    labels = rng.random((n_prompts, n_resp)) < 0.5
    mac_b, nd_b = P.equal_prompt_macro_auroc(scores, labels)
    mac_f, nd_f = P.equal_prompt_macro_auroc(
        scores.ravel(), labels.ravel(), np.repeat(np.arange(n_prompts), n_resp)
    )
    assert nd_b == nd_f
    assert abs(mac_b - mac_f) < 1e-12


def test_batched_flat_labels_match_per_row_calls():
    rng = np.random.default_rng(4)
    pid = np.repeat(np.arange(5), 4)
    scores = rng.normal(size=20)
    label_stack = rng.random((6, 20)) < 0.4
    mb, cb = P.equal_prompt_macro_auroc(scores, label_stack, pid)
    assert mb.shape == (6,) and cb.shape == (6,)
    for i in range(6):
        mi, ci = P.equal_prompt_macro_auroc(scores, label_stack[i], pid)
        assert cb[i] == ci
        assert (np.isnan(mb[i]) and np.isnan(mi)) or abs(mb[i] - mi) < 1e-12


def test_permutation_stays_within_prompt():
    rng = np.random.default_rng(5)
    pid = np.array(["p1"] * 5 + ["p2"] * 7 + ["p3"] * 3)
    labels = rng.random(15) < 0.4
    perm = P.permute_labels_within_prompt(labels, pid, np.random.default_rng(6), n_perm=200)
    assert perm.shape == (200, 15)
    for g in ("p1", "p2", "p3"):
        m = pid == g
        # per-prompt label multiset preserved in EVERY draw => nothing crossed prompts
        assert (perm[:, m].sum(axis=1) == labels[m].sum()).all()
    assert (perm != labels).any()  # labels actually move
    single = P.permute_labels_within_prompt(labels, pid, np.random.default_rng(7))
    assert single.shape == labels.shape and single.dtype == labels.dtype


def test_permutation_is_uniform_within_group():
    # a 2-element group must realize BOTH orders across many draws
    pid = np.array(["a", "a", "b"])
    labels = np.array([True, False, True])
    perm = P.permute_labels_within_prompt(labels, pid, np.random.default_rng(8), n_perm=200)
    first = perm[:, 0]
    assert first.any() and not first.all()  # both assignments appear
    assert perm[:, 2].all()  # the singleton group can never change


# ---------------------------------------------------------------------------
# Deliverable B — discordance projection + bounds.
# ---------------------------------------------------------------------------
def test_projection_10_vs_30_strictly_larger_for_interior_p():
    k = np.arange(1, 10)
    n = np.full(9, 10)
    p10 = P.project_discordance_plugin(k, n, 10)
    p30 = P.project_discordance_plugin(k, n, 30)
    assert (p30 > p10).all()  # discordance probability increases in draws
    boundary = P.project_discordance_plugin(np.array([0, 10]), np.array([10, 10]), 30)
    assert boundary.max() == 0.0  # plug-in degenerates to certainty at p in {0,1}
    jeff = P.project_discordance_jeffreys(np.array([0, 10]), np.array([10, 10]), 30)
    assert (jeff > 0).all()  # the shrunk projection does not
    interior = P.project_discordance_jeffreys(k, n, 30)
    assert ((interior > 0) & (interior < 1)).all()


def test_clopper_pearson_lower_bound():
    from scipy import stats as sps

    assert P.clopper_pearson_lower(0, 5) == 0.0
    got = P.clopper_pearson_lower(3, 5)
    assert abs(got - float(sps.beta.ppf(0.05, 3, 3))) < 1e-12
    # monotone in x; strictly below the point estimate
    lbs = [P.clopper_pearson_lower(x, 10) for x in range(11)]
    import itertools

    assert all(b1 <= b2 for b1, b2 in itertools.pairwise(lbs))
    assert P.clopper_pearson_lower(5, 10) < 0.5
    with pytest.raises(ValueError):
        P.clopper_pearson_lower(6, 5)


def test_measure_discordance_report():
    prof = P.RowLabelProfile(row=JUDGED_ROW, judged=True)
    cell = P.expected_cells(JUDGED_ROW)[0]
    prof.cells[cell] = [(0, 10), (3, 10), (10, 10), (5, 10), (1, 10)]  # 3 of 5 discordant
    rep = P.measure_discordance({JUDGED_ROW: prof}, seed=0)
    c = rep["rows"][JUDGED_ROW]["cells"][cell]
    assert c["m_prompts"] == 5 and c["x_discordant_10draw"] == 3
    assert abs(c["raw_rate_10draw"] - 0.6) < 1e-12
    assert abs(c["cp_lower95_10draw"] - P.clopper_pearson_lower(3, 5)) < 1e-12
    # sizing bound is the max rule; requirement = ceil(20 / bound)
    lb = max(c["cp_lower95_10draw"], c["credible_lower05_30draw_jeffreys"])
    assert abs(c["sizing_lower_bound"] - lb) < 1e-12
    assert c["n_required_for_target"] == int(np.ceil(20 / lb))
    # monotonicity holds between PLUG-IN projections at fixed p-hat (1-p^m-(1-p)^m
    # increasing in m) — NOT vs the raw empirical indicator rate: a discordant
    # prompt with p-hat=0.1 contributes 1.0 to the raw rate but only ~0.958 to
    # the 30-draw plug-in, so the mean plug-in can dip below the raw rate.
    ks = np.array([0, 3, 10, 5, 1], dtype=np.int64)
    ns = np.full(5, 10, dtype=np.int64)
    plugin10 = float(P.project_discordance_plugin(ks, ns, 10).mean())
    assert c["projected_rate_30draw_plugin"] >= plugin10 - 1e-12


def test_measure_discordance_zero_discordant_cell_unbounded():
    prof = P.RowLabelProfile(row=JUDGED_ROW, judged=True)
    cell = P.expected_cells(JUDGED_ROW)[0]
    prof.cells[cell] = [(0, 10), (10, 10)]  # unanimous prompts only
    rep = P.measure_discordance({JUDGED_ROW: prof}, seed=0)
    c = rep["rows"][JUDGED_ROW]["cells"][cell]
    assert c["x_discordant_10draw"] == 0
    assert c["cp_lower95_10draw"] == 0.0
    assert c["n_required_for_target"] is None  # unbounded — never a silent default


# ---------------------------------------------------------------------------
# Deliverable C — clustered power simulation + ledger resume.
# ---------------------------------------------------------------------------
def _tiny_pools(n_cells: int = 2, n_prompts: int = 6) -> dict[str, list[tuple[int, int]]]:
    rng = np.random.default_rng(9)
    return {
        f"cell{i}": [(int(k), 10) for k in rng.integers(2, 9, n_prompts)] for i in range(n_cells)
    }


def test_simulate_power_size_under_null_and_power_under_strong_effect():
    pools = _tiny_pools()
    # effect 0.5 => zero shift => the test's rejection rate ~= its exact size.
    # alpha 0.2 with 19 permutations: p = (1+k)/20 <= 0.2 iff k <= 3 (size 4/20).
    null = P.simulate_power(
        pools, 6, 0.5000001, alpha=0.2, n_reps=80, n_perm=19, m_resp=10, seed=10
    )
    assert 0.05 <= null["power"] <= 0.40  # ~0.2 +/- MC noise (SE ~= 0.045)
    strong = P.simulate_power(pools, 8, 0.90, alpha=0.2, n_reps=40, n_perm=19, m_resp=10, seed=11)
    assert strong["power"] >= 0.9
    assert strong["mean_discordant_prompts"] > 0
    # binormal calibration: the realized macro statistic sits near the target
    assert abs(strong["mean_stat"] - 0.90) < 0.05


def test_simulate_power_unrejectable_configuration_raises():
    with pytest.raises(P.PowerInputError, match="unrejectable"):
        P.simulate_power(
            _tiny_pools(), 4, 0.6, alpha=0.05 / 11, n_reps=4, n_perm=99, m_resp=10, seed=0
        )


def test_simulate_power_empty_pool_raises():
    with pytest.raises(P.PowerInputError, match="empty prompt pool"):
        P.simulate_power({"c": []}, 4, 0.6, alpha=0.2, n_reps=4, n_perm=19, m_resp=10, seed=0)


def test_ledger_resume_skips_completed_and_recomputes_on_param_change(tmp_path):
    reg = P.PowerRegistry(n_replicates=3, n_permutations=219)
    ledger = P.PowerLedger(tmp_path / "units.jsonl")
    counter = P._UnitCounter(cap=3)
    kwargs = dict(
        row=JUDGED_ROW,
        cell_pools=_tiny_pools(1, 4),
        n_prompts_per_cell=3,
        effect_auroc=0.7,
        reg=reg,
        n_reps=3,
        n_perm=219,
        seed=0,
        prof_sha="deadbeef",
        purpose="bisection",
    )
    rec1 = P._run_power_unit(ledger, counter, **kwargs)
    rec2 = P._run_power_unit(ledger, counter, **kwargs)
    assert rec2 == rec1  # resume-skip returned the persisted record
    lines = (tmp_path / "units.jsonl").read_text().strip().splitlines()
    assert len(lines) == 1  # nothing recomputed
    # a fresh ledger over the same file also resumes (durable across processes)
    ledger2 = P.PowerLedger(tmp_path / "units.jsonl")
    rec3 = P._run_power_unit(ledger2, P._UnitCounter(cap=1), **kwargs)
    assert rec3 == rec1
    # changing ANY generating parameter (seed) recomputes
    rec4 = P._run_power_unit(ledger, counter, **{**kwargs, "seed": 1})
    assert rec4["key"] != rec1["key"]
    assert len((tmp_path / "units.jsonl").read_text().strip().splitlines()) == 2


def test_unit_key_requires_every_generating_parameter():
    with pytest.raises(P.PowerInputError, match="missing generating parameters"):
        P.PowerLedger.unit_key(row="evil", seed=0)


def test_select_production_n_floor_and_binding(tmp_path):
    reg = P.PowerRegistry(
        n_replicates=6,
        n_permutations=219,
        power_curve_effects=(0.70,),
        prompts_per_cell_floor=4,
        bisection_cap=16,
        primary_effect_auroc=0.70,
    )
    profiles = P.synthetic_profile([JUDGED_ROW, OBJECTIVE_ROW], seed=3)
    disc = P.measure_discordance(profiles, seed=0)
    ledger = P.PowerLedger(tmp_path / "units.jsonl")
    sel = P.select_production_n(profiles, disc, ledger, reg=reg, n_reps=6, n_perm=219, seed=0)
    assert sel["registered_match"] is False  # non-registered sizes => gate must FAIL
    gate = P._gate_power(sel, tmp_path)
    assert gate.status == P.GATE_FAIL and "non-registered" in gate.detail
    if sel["status"] == "measured":
        assert sel["n_common"] >= reg.prompts_per_cell_floor
        assert sel["binding_discordance_cell"] is not None
        assert sel["binding_power_row"] is not None
        assert set(sel["power_curve_at_n_common"]) == {JUDGED_ROW, OBJECTIVE_ROW}
    else:
        assert sel["n_common"] is None  # not-estimable is explicit, never defaulted


# ---------------------------------------------------------------------------
# Deliverable B loaders — production-schema fixtures, drop-never-coerce.
# ---------------------------------------------------------------------------
def _judge_cell_fixture(cell: str, verdicts: dict) -> dict:
    return {
        "schema": J.JUDGE_SCHEMA,
        "row": JUDGED_ROW,
        "split": "pilot",
        "cell": cell,
        "verdicts": verdicts,
        "counters": {},
        "n_units": len(verdicts),
        "n_scored": sum(1 for v in verdicts.values() if v["judge_status"] == "scored"),
        "n_human_adjudication": sum(1 for v in verdicts.values() if v["judge_status"] != "scored"),
        "parse_fail_rate": 0.0,
        "stop_reason_tally": {"end_turn": 5 * len(verdicts)},
        "plan_gate": {"parse_fail_lt_threshold": True, "zero_max_tokens_stops": True},
    }


def test_judge_profile_loader_drop_never_coerce(tmp_path):
    cell = P.expected_cells(JUDGED_ROW)[0]
    verdicts = {}
    # item i0: 2 scored (1 positive), 1 human_adjudication draw-starved unit
    for ridx, (status, label) in enumerate(
        [("scored", True), ("scored", False), ("human_adjudication", None)]
    ):
        uid = f"i0#r{ridx:02d}"
        verdicts[uid] = {
            "unit_id": uid,
            "item_id": "i0",
            "response_index": ridx,
            "judge_status": status,
            "binary_label": label,
            "median_score": 80.0 if label else None,
        }
    # item i1: fully unlabeled (all human_adjudication) => EXCLUDED, counted
    verdicts["i1#r00"] = {
        "unit_id": "i1#r00",
        "item_id": "i1",
        "response_index": 0,
        "judge_status": "human_adjudication",
        "binary_label": None,
        "median_score": None,
    }
    d = tmp_path / "judge" / "pilot" / JUDGED_ROW
    d.mkdir(parents=True)
    (d / f"{cell}.json").write_text(json.dumps(_judge_cell_fixture(cell, verdicts)))
    profiles = P.load_pilot_label_profile(tmp_path, "pilot", [JUDGED_ROW])
    prof = profiles[JUDGED_ROW]
    assert prof.cells[cell] == [(1, 2)]  # only scored draws enter (k, n)
    assert prof.n_unlabeled_prompts == 1  # i1 counted, never coerced
    assert len(prof.missing_cells) == 11  # the other cells have no artifacts


def test_judge_profile_loader_rejects_foreign_schema(tmp_path):
    cell = P.expected_cells(JUDGED_ROW)[0]
    d = tmp_path / "judge" / "pilot" / JUDGED_ROW
    d.mkdir(parents=True)
    (d / f"{cell}.json").write_text(json.dumps({"schema": "other", "verdicts": {}}))
    with pytest.raises(P.PowerInputError, match="schema"):
        P.load_pilot_label_profile(tmp_path, "pilot", [JUDGED_ROW])


def test_objective_profile_loader_counts_only_labeled(tmp_path):
    cell = P.expected_cells(OBJECTIVE_ROW)[0]
    d = tmp_path / "objective_labels" / "pilot"
    d.mkdir(parents=True)
    rows = [
        {
            "schema": "i2658-objective-labels-v1",
            "manifest": {"prompt_id": "q0", "response_index": 0},
            "label": True,
            "status": "labeled",
        },
        {
            "schema": "i2658-objective-labels-v1",
            "manifest": {"prompt_id": "q0", "response_index": 1},
            "label": False,
            "status": "labeled",
        },
        {
            "schema": "i2658-objective-labels-v1",
            "manifest": {"prompt_id": "q0", "response_index": 2},
            "label": None,
            "status": "malformed",
        },
    ]
    (d / f"{cell}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    profiles = P.load_pilot_label_profile(tmp_path, "pilot", [OBJECTIVE_ROW])
    assert profiles[OBJECTIVE_ROW].cells[cell] == [(1, 2)]  # malformed dropped + counted


# ---------------------------------------------------------------------------
# Deliverable D — cost report honesty.
# ---------------------------------------------------------------------------
def test_projection_never_reported_as_measured(tmp_path):
    rep = P.cost_report(tmp_path, tmp_path, "pilot", n_common=None)
    gpu = rep["gpu_hours"]["measured_pilot_gpu_h"]
    assert gpu["value"] is None and gpu["basis"] == "not-estimable"
    assert P.PILOT_TIMING_REL in gpu["missing_artifact"]  # missing artifact NAMED
    proj = rep["api"]["projected_pilot_judge_draws"]
    assert proj["value"] == 24_000  # 8 judged traits x 600 answers x 5 draws
    assert proj["basis"] == "projected"  # NEVER quoted as realized
    assert rep["api"]["measured_dollars"]["basis"] == "not-estimable"
    assert rep["human_annotation"]["required_minimum_adjudications"]["basis"] == "projected"
    assert rep["human_annotation"]["required_minimum_adjudications"]["value"] == 1600
    assert rep["envelope"]["within_envelope"] is None  # unknown != within


def test_cost_report_measured_path_and_envelope(tmp_path):
    timing = tmp_path / P.PILOT_TIMING_REL
    timing.parent.mkdir(parents=True)
    timing.write_text(json.dumps({"wall_hours": 0.75, "gpu_count": 8, "n_responses": 6600}))
    rep = P.cost_report(tmp_path, tmp_path, "pilot", n_common=30)
    assert rep["gpu_hours"]["measured_pilot_gpu_h"] == {
        "value": 6.0,
        "basis": "measured",
        "artifact": str(timing),
    }
    proj = rep["gpu_hours"]["projected_production_gpu_h"]
    assert proj["basis"] == "projected"
    # marginal 6/6600 GPU-h/response x (11*12*30*30*2 = 237,600 responses) = 216
    assert abs(proj["value"] - 6.0 / 6600 * 237_600) < 1e-9
    env = rep["envelope"]
    assert env["within_envelope"] is False  # 216 + 6 > 80 => honest FAIL, not a default
    assert env["margin_gpu_h"] < 0


# ---------------------------------------------------------------------------
# Reliability gates (plan section 3).
# ---------------------------------------------------------------------------
def test_reliability_not_estimable_on_absent_labels(tmp_path):
    rel = P.reliability_gates(tmp_path)
    assert rel["status"] == P.GATE_NOT_ESTIMABLE
    assert P.HUMAN_AUDIT_REL in rel["missing_artifact"]


def _audit_fixture(n_pos: int, n_neg: int, judge_flip_every: int = 50) -> dict:
    rows = []
    for i in range(n_pos + n_neg):
        pos = i < n_pos
        prob = 90.0 if pos else 10.0
        judge = pos if (i % judge_flip_every) else (not pos)  # a few judge errors
        rows.append(
            {
                "row": JUDGED_ROW,
                "item_id": f"i{i}",
                "response_index": 0,
                "rater_a_prob": prob,
                "rater_b_prob": prob,
                "judge_binary": judge,
            }
        )
    return {"schema": "i2658-human-adjudication-v1", "rows": rows}


def test_reliability_gates_pass_on_agreeing_fixture(tmp_path):
    audit = tmp_path / P.HUMAN_AUDIT_REL
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps(_audit_fixture(240, 240)))
    rel = P.reliability_gates(tmp_path)
    trait = rel["per_trait"][JUDGED_ROW]
    assert rel["status"] == P.GATE_PASS
    assert trait["kappa"] == 1.0  # raters agree exactly
    assert trait["icc_2_1"] > 0.99
    assert trait["sensitivity_lower95"] >= 0.80
    assert trait["sensitivity_lower95"] < trait["sensitivity"]  # a bound, not the point


def test_reliability_gates_not_estimable_below_sizing_floor(tmp_path):
    audit = tmp_path / P.HUMAN_AUDIT_REL
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps(_audit_fixture(40, 240)))  # < 100 positives
    rel = P.reliability_gates(tmp_path)
    assert rel["status"] == P.GATE_NOT_ESTIMABLE
    assert ">= 100" in rel["per_trait"][JUDGED_ROW]["detail"]


def test_icc_and_kappa_known_fixtures():
    perfect = np.column_stack([np.arange(10.0), np.arange(10.0)])
    assert P.icc_2_1(perfect) > 0.999
    rng = np.random.default_rng(12)
    noise = np.column_stack([rng.normal(size=200), rng.normal(size=200)])
    assert abs(P.icc_2_1(noise)) < 0.2
    with pytest.raises(ValueError):
        P.icc_2_1(np.ones((5, 1)))
    ss = P.sensitivity_specificity_lower(
        np.array([True, True, False, False]), np.array([True, True, False, False])
    )
    assert ss["sensitivity"] == 1.0 and ss["specificity"] == 1.0
    assert abs(ss["sensitivity_lower95"] - 0.05 ** (1 / 2)) < 1e-12  # CP at x=n=2


# ---------------------------------------------------------------------------
# Deliverable E — verdict aggregation: NOT-ESTIMABLE never collapses to PASS.
# ---------------------------------------------------------------------------
def test_not_estimable_never_becomes_pass(tmp_path):
    cost = P.cost_report(tmp_path, tmp_path, "pilot", n_common=None)
    verdict = P.evaluate_gates(tmp_path, tmp_path, "pilot", None, None, cost)
    assert verdict["verdict"] == "PARK"
    by_id = {g["gate_id"]: g for g in verdict["gates"]}
    # the plan section 10 human dependency, encoded honestly: no adjudications
    # on disk => NOT-ESTIMABLE => PARK (the correct pre-audit output)
    assert by_id["human_audit_feasibility"]["status"] == P.GATE_NOT_ESTIMABLE
    assert "human_audit_feasibility" in verdict["blockers"]
    for g in verdict["gates"]:
        if g["status"] != P.GATE_PASS:
            assert g["gate_id"] in verdict["blockers"]
    assert len(verdict["gates"]) == 10  # one entry per plan section 8 pilot gate


def test_gate_status_vocabulary_is_closed():
    with pytest.raises(ValueError, match="invalid gate status"):
        P.Gate("x", "d", "MAYBE", None, "t", "a")


def test_import_check_mode_runs():
    assert P.main(["--import-check"]) == 0


def test_registered_constants_are_frozen_and_consistent():
    reg = P.REGISTERED
    assert reg.power_target == 0.80
    assert abs(reg.alpha_worst_case - 0.05 / 11) < 1e-15
    assert reg.primary_effect_auroc == 0.60
    assert reg.power_curve_effects == (0.55, 0.60, 0.65, 0.70)
    # B=659 makes the Holm worst case EXACTLY reachable: (1+2)/660 == 1/220
    assert (1 + 2) / (reg.n_permutations + 1) <= reg.alpha_worst_case + 1e-15
    assert 1 / (reg.n_permutations + 1) <= reg.alpha_worst_case
    import dataclasses

    with pytest.raises(dataclasses.FrozenInstanceError):
        reg.power_target = 0.9  # frozen dataclass


def test_synthetic_profile_is_smoke_only_labeled():
    prof = P.synthetic_profile([JUDGED_ROW], seed=0)
    assert prof[JUDGED_ROW].artifact_dir == "SYNTHETIC (smoke)"
    assert len(prof[JUDGED_ROW].cells) == C.PILOT.cells_per_row
