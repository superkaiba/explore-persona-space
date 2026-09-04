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


def _timing_fixture(
    wall_hours: float = 3.0,
    gpu_count: int = 8,
    gen_marginal: float = 0.14,
    capture_rate: float = 35.0,
    overhead_h: float = 0.135,
) -> dict:
    """A v2 pilot-timing artifact shaped like issue2658_pilot_timing.py output."""
    return {
        "schema": "i2658-pilot-timing-v2",
        "wall_hours": wall_hours,
        "gpu_count": gpu_count,
        "n_responses": 6290,
        "fixed_overhead_hours": overhead_h,
        "pod_wall_hours_all_in": wall_hours,
        "gpu_hours_all_in": wall_hours * gpu_count,
        "gen_marginal_s_per_response_per_gpu": gen_marginal,
        "gen_engine_init_s_per_shard": {"generate_shard00": overhead_h * 3600},
        "capture_rows_per_s_per_gpu": capture_rate,
        "capture_model_load_s_per_shard": {"value": None, "basis": "not-measured"},
        "shards_used_for_gen_rate": ["generate_shard00"],
        "crash_fix_rounds_note": "synthetic: 1 start, 0 crash-fix restarts",
    }


def test_cost_report_measured_marginal_path_and_envelope(tmp_path):
    timing = tmp_path / P.PILOT_TIMING_REL
    timing.parent.mkdir(parents=True)
    timing.write_text(json.dumps(_timing_fixture()))
    rep = P.cost_report(tmp_path, tmp_path, "pilot", n_common=30)
    meas = rep["gpu_hours"]["measured_pilot_gpu_h"]
    assert meas["value"] == 24.0 and meas["basis"].startswith("measured")
    proj = rep["gpu_hours"]["projected_production_gpu_h_measured_marginal"]
    assert proj["basis"].startswith("projected")
    # v5 A4 measured-marginal formula at N=30 (237,600 production responses):
    n_prod = 11 * 12 * 30 * 30 * 2
    gen = n_prod * 0.14 / 3600 + 2 * 0.135
    cap = n_prod / 35.0 / 3600
    assert abs(proj["value"] - (gen + cap)) < 1e-9
    assert "LOWER BOUND" in proj["caveat"]
    env = rep["envelope"]
    # pilot 24 GPU-h >> the 8 GPU-h ceiling, yet the total (~35.4) is inside the
    # 80 GPU-h kill criterion => within_envelope True, deviation REPORTED (v5 A4)
    assert env["within_envelope"] is True
    assert env["pilot_within_ceiling"] is False
    dev = env["pilot_ceiling_deviation"]
    assert dev["measured_pilot_gpu_h"] == 24.0 and dev["ratio"] == 3.0
    assert "synthetic: 1 start" in dev["decomposition"]
    gate = P._gate_cost(rep)
    assert gate.status == P.GATE_PASS
    assert gate.measured["pilot_ceiling_deviation"]["ratio"] == 3.0


def test_cost_gate_fails_only_on_the_80h_kill_criterion(tmp_path):
    timing = tmp_path / P.PILOT_TIMING_REL
    timing.parent.mkdir(parents=True)
    timing.write_text(json.dumps(_timing_fixture(gen_marginal=1.2)))
    rep = P.cost_report(tmp_path, tmp_path, "pilot", n_common=30)
    env = rep["envelope"]
    assert env["projected_total_gpu_h"] > 80.0
    assert env["within_envelope"] is False and env["margin_gpu_h"] < 0
    gate = P._gate_cost(rep)
    assert gate.status == P.GATE_FAIL and "kill criterion" in gate.detail


def test_cost_report_rejects_legacy_timing_artifact(tmp_path):
    # a v1-shaped artifact (the four legacy fields only) is REFUSED, never
    # silently projected from — the v5 A4 measured components are required
    timing = tmp_path / P.PILOT_TIMING_REL
    timing.parent.mkdir(parents=True)
    timing.write_text(json.dumps({"wall_hours": 0.75, "gpu_count": 8, "n_responses": 6600}))
    with pytest.raises(P.PowerInputError, match="missing field"):
        P.cost_report(tmp_path, tmp_path, "pilot", n_common=30)


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
    profiles = P.load_pilot_label_profile(tmp_path, "pilot")  # no artifacts on disk
    verdict = P.evaluate_gates(tmp_path, tmp_path, "pilot", profiles, None, None, cost)
    assert verdict["verdict"] == "PARK"
    by_id = {g["gate_id"]: g for g in verdict["gates"]}
    # the plan section 10 human dependency, encoded honestly: no adjudications
    # on disk => NOT-ESTIMABLE => PARK (the correct pre-audit output)
    assert by_id["human_audit_feasibility"]["status"] == P.GATE_NOT_ESTIMABLE
    assert "human_audit_feasibility" in verdict["blockers"]
    # nothing to cross-check => NOT-ESTIMABLE, never a vacuous PASS
    assert by_id["profile_freshness"]["status"] == P.GATE_NOT_ESTIMABLE
    for g in verdict["gates"]:
        if g["status"] != P.GATE_PASS:
            assert g["gate_id"] in verdict["blockers"]
    # plan section 8 pilot gates + the profile-freshness cross-check
    assert len(verdict["gates"]) == 11


def _full_coverage_selection(n_common: int = 48) -> dict:
    """A selection artifact shaped like a REGISTERED full-row run (gate PASS)."""
    return {
        "registered_match": True,
        "status": "measured",
        "n_common": n_common,
        "rows_simulated": sorted(C.ROW_IDS),
        "rows_dead": [],
        "rows_with_declared_exclusions": [],
        "rows_missing_labels_undocumented": [],
        "per_row_power_n": {},
        "cells_not_estimable_zero_discordance": [],
        "cells_estimable": [],
        "binding_discordance_cell": {"row": "evil", "cell": "evil__x__y", "n_required": 23},
        "binding_power_row": {"row": "evil", "n_required": 30},
        "profile_sha256": "0" * 64,
    }


def test_gate_power_fails_on_row_subset_even_at_registered_sizes(tmp_path):
    # The round-1 live-probe shape: a selection exactly as `--phase power
    # --rows evil` produces at REGISTERED constants (rows_simulated=["evil"],
    # status "measured", registered_match True) must NOT authorize a launch —
    # its n_common is sized on ONE row, not the max over all 11.
    sel = _full_coverage_selection()
    sel["rows_simulated"] = ["evil"]
    gate = P._gate_power(sel, tmp_path)
    assert gate.status == P.GATE_FAIL
    assert "missing rows" in gate.detail
    for row in sorted(set(C.ROW_IDS) - {"evil"}):
        assert row in gate.detail  # every missing row is NAMED


def test_gate_power_passes_on_full_row_coverage(tmp_path):
    # the other direction: full registered-universe coverage still PASSes
    gate = P._gate_power(_full_coverage_selection(), tmp_path)
    assert gate.status == P.GATE_PASS
    assert gate.measured["n_common"] == 48


def test_gate_profile_freshness_names_mismatched_shas(tmp_path):
    live = P.synthetic_profile([JUDGED_ROW], seed=0)
    stale = P.synthetic_profile([JUDGED_ROW], seed=7)  # different (k, n) pools
    disc_live = P.measure_discordance(live, seed=0)
    disc_stale = P.measure_discordance(stale, seed=0)
    live_sha = P.profile_fingerprint(live)
    assert disc_stale["profile_sha256"] != live_sha

    # matched artifact <-> live labels: PASS
    g = P._gate_profile_freshness(live, disc_live, None, tmp_path)
    assert g.status == P.GATE_PASS

    # stale artifact: FAIL naming BOTH shas
    g2 = P._gate_profile_freshness(live, disc_stale, None, tmp_path)
    assert g2.status == P.GATE_FAIL
    assert disc_stale["profile_sha256"] in g2.detail and live_sha in g2.detail

    # selection/discordance pairwise mismatch is ALSO named
    sel = dict(_full_coverage_selection(), profile_sha256="f" * 64)
    g3 = P._gate_profile_freshness(live, disc_stale, sel, tmp_path)
    assert g3.status == P.GATE_FAIL and "mixed-generation" in g3.detail

    # an artifact with no fingerprint at all is fail-closed, never skipped
    g4 = P._gate_profile_freshness(live, disc_live, {"n_common": 30}, tmp_path)
    assert g4.status == P.GATE_FAIL and "no profile_sha256" in g4.detail

    # nothing to cross-check: NOT-ESTIMABLE (never a vacuous PASS)
    g5 = P._gate_profile_freshness(live, None, None, tmp_path)
    assert g5.status == P.GATE_NOT_ESTIMABLE


def _profile_fixture(tmp_path, name: str, pairs, rows=None) -> Path:
    """Write a {row: {cell: [[k, n], ...]}} --profile-json fixture (one cell/row)."""
    body = {
        row: {f"{row}|cellA": [list(p) for p in pairs]}
        for row in (rows if rows is not None else C.ROW_IDS)
    }
    p = tmp_path / name
    p.write_text(json.dumps(body))
    return p


def test_sharded_rows_then_full_row_selection_end_to_end(tmp_path):
    """The recorded P2-P3 dispatch shape: shard by --rows across pods (shared
    ledger), then ONE final full-row invocation selects N. Both directions:
    (a) a shard's subset selection can never authorize a launch; (b) the final
    full-row invocation resumes every bisection unit from the shared ledger and
    its selection covers the full registered row universe."""
    fx = _profile_fixture(tmp_path, "fxA.json", pairs=((3, 10), (5, 10), (2, 10)))
    out = tmp_path / "out"
    ap = P.build_argparser()
    base = [
        "--phase",
        "power",
        "--out-root",
        str(out),
        "--profile-json",
        str(fx),
        "--reps",
        "8",
        "--n-perm",
        "219",
    ]
    ledger_path = out / "power" / "power_units.jsonl"

    def _n_bisection_lines() -> int:
        recs = [json.loads(x) for x in ledger_path.read_text().strip().splitlines()]
        return sum(1 for r in recs if r["purpose"] == "bisection")

    # shard 1: one row
    assert P.run(ap.parse_args([*base, "--rows", "evil"])) == 0
    sel1 = json.loads((out / "power" / "production_n.json").read_text())
    assert sel1["rows_simulated"] == ["evil"]
    n_shard1 = _n_bisection_lines()
    assert n_shard1 > 0

    # direction (a): the shard artifact, even at registered sizes, cannot LAUNCH
    probe = dict(sel1, registered_match=True, status="measured", n_common=48)
    gate = P._gate_power(probe, out)
    assert gate.status == P.GATE_FAIL and "missing rows" in gate.detail
    assert "sycophancy" in gate.detail

    # shard 2: the remaining ten rows (parallel-pod shape, same shared ledger)
    rest = [r for r in C.ROW_IDS if r != "evil"]
    assert P.run(ap.parse_args([*base, "--rows", *rest])) == 0
    n_shards = _n_bisection_lines()
    assert n_shards > n_shard1

    # direction (b): the final full-row invocation
    assert P.run(ap.parse_args(base)) == 0
    assert _n_bisection_lines() == n_shards  # every bisection unit resume-skipped
    sel = json.loads((out / "power" / "production_n.json").read_text())
    assert sel["rows_simulated"] == sorted(C.ROW_IDS)
    # and the full-row selection does NOT trip the coverage check
    g_full = P._gate_power(dict(sel, registered_match=True), out)
    assert "missing rows" not in (g_full.detail or "")


def test_standalone_gate_phase_cross_checks_profile_freshness(tmp_path):
    """Fix-2 shape: judge cells regenerated post-adjudication, then `--phase
    gate` re-run standalone — the stale discordance/production-N artifacts must
    FAIL the freshness gate, naming the shas; a matched re-gate PASSes it."""
    rows = ["evil", "correctness_math"]
    fx_a = _profile_fixture(tmp_path, "fxA.json", pairs=((3, 10), (5, 10), (2, 10)), rows=rows)
    fx_b = _profile_fixture(tmp_path, "fxB.json", pairs=((1, 10), (9, 10), (4, 10)), rows=rows)
    out = tmp_path / "out"
    ap = P.build_argparser()
    base = ["--out-root", str(out), "--reps", "4", "--n-perm", "219", "--rows", *rows]

    # full run from labels A: artifacts fingerprint A; same-run gate is FRESH
    assert P.run(ap.parse_args([*base, "--phase", "all", "--profile-json", str(fx_a)])) == 0
    v1 = json.loads((out / "power" / "gate_verdict.json").read_text())
    fresh1 = {g["gate_id"]: g for g in v1["gates"]}["profile_freshness"]
    assert fresh1["status"] == P.GATE_PASS

    # labels regenerated (B) -> standalone gate re-run reads STALE artifacts
    assert P.run(ap.parse_args([*base, "--phase", "gate", "--profile-json", str(fx_b)])) == 0
    v2 = json.loads((out / "power" / "gate_verdict.json").read_text())
    fresh2 = {g["gate_id"]: g for g in v2["gates"]}["profile_freshness"]
    assert fresh2["status"] == P.GATE_FAIL
    assert "profile_freshness" in v2["blockers"] and v2["verdict"] == "PARK"
    stale_sha = json.loads((out / "power" / "production_n.json").read_text())["profile_sha256"]
    assert stale_sha in fresh2["detail"]  # the stale sha is NAMED
    assert fresh2["measured"]["live_profiles"] in fresh2["detail"]  # and the live one

    # matched re-gate (labels A again): freshness PASSes
    assert P.run(ap.parse_args([*base, "--phase", "gate", "--profile-json", str(fx_a)])) == 0
    v3 = json.loads((out / "power" / "gate_verdict.json").read_text())
    fresh3 = {g["gate_id"]: g for g in v3["gates"]}["profile_freshness"]
    assert fresh3["status"] == P.GATE_PASS


def test_phase_all_with_stale_selection_on_disk_cannot_certify(tmp_path):
    """Fix-2 second shape: `--phase all` where labels exist but carry no cells
    silently loads a STALE production_n.json from disk (run():2033) — the
    freshness gate must refuse to certify it."""
    rows = ["evil", "correctness_math"]
    fx_a = _profile_fixture(tmp_path, "fxA.json", pairs=((3, 10), (5, 10), (2, 10)), rows=rows)
    fx_empty = tmp_path / "fxE.json"
    fx_empty.write_text(json.dumps({row: {} for row in rows}))  # rows, ZERO cells
    out = tmp_path / "out"
    ap = P.build_argparser()
    base = ["--out-root", str(out), "--reps", "4", "--n-perm", "219", "--rows", *rows]
    assert P.run(ap.parse_args([*base, "--phase", "all", "--profile-json", str(fx_a)])) == 0
    assert P.run(ap.parse_args([*base, "--phase", "all", "--profile-json", str(fx_empty)])) == 0
    v = json.loads((out / "power" / "gate_verdict.json").read_text())
    fresh = {g["gate_id"]: g for g in v["gates"]}["profile_freshness"]
    assert fresh["status"] == P.GATE_FAIL
    assert "stale artifact" in fresh["detail"]
    assert v["verdict"] == "PARK"


def test_gate_status_vocabulary_is_closed():
    with pytest.raises(ValueError, match="invalid gate status"):
        P.Gate("x", "d", "MAYBE", None, "t", "a")
    # plan v6 A7: AMENDED is a member of the closed vocabulary (never PASS)
    assert P.Gate("x", "d", P.GATE_AMENDED, None, "t", "a").status == "AMENDED"


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


# ---------------------------------------------------------------------------
# Round 12 (plan v5): waiver / declared exclusions / zero-discordance / timing
# / draw-count / stager. All offline + synthetic.
# ---------------------------------------------------------------------------
def _waiver_fixture() -> dict:
    return {
        "schema": P.WAIVER_SCHEMA,
        "ruling_event": {
            "kind": "epm:clarify-answers",
            "version": 1,
            "ts": "2026-09-03T20:39:25Z",
            "by": "thomas-via-watcher-7d7549",
        },
        "ruling_verbatim": "just trust the judge",
        "scope": {
            "banks": ["dev", "test"],
            "gates": ["human_audit_feasibility", "label_reliability"],
        },
        "disclosure": "SYNTHETIC-DISCLOSURE: judge labels, no human validation.",
        "plan_version": "v5",
    }


def _write_waiver(tmp_path, body) -> None:
    p = tmp_path / P.HUMAN_AUDIT_WAIVER_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body))


def test_waiver_absent_present_malformed(tmp_path):
    # absent: unchanged NOT-ESTIMABLE
    rel = P.reliability_gates(tmp_path)
    assert rel["status"] == P.GATE_NOT_ESTIMABLE and rel["per_trait"] == {}
    # present + valid: WAIVED for every judged row, disclosure verbatim
    _write_waiver(tmp_path, _waiver_fixture())
    rel = P.reliability_gates(tmp_path)
    judged = [r for r in C.ROW_IDS if C.CONSTRUCTS[r].judge_scored]
    assert rel["status"] == P.GATE_WAIVED
    assert sorted(rel["per_trait"]) == sorted(judged)
    for v in rel["per_trait"].values():
        assert v["status"] == P.GATE_WAIVED and "SYNTHETIC-DISCLOSURE" in v["detail"]
    assert rel["waiver"]["ruling_verbatim"] == "just trust the judge"
    # malformed: RAISES, never degrades to WAIVED or NOT-ESTIMABLE
    bad = _waiver_fixture()
    bad["scope"]["banks"] = ["dev"]
    _write_waiver(tmp_path, bad)
    with pytest.raises(P.PowerInputError, match="malformed waiver"):
        P.reliability_gates(tmp_path)
    bad2 = _waiver_fixture()
    bad2.pop("disclosure")
    _write_waiver(tmp_path, bad2)
    with pytest.raises(P.PowerInputError, match="disclosure"):
        P.reliability_gates(tmp_path)


def test_waiver_ignored_when_real_adjudications_exist(tmp_path):
    _write_waiver(tmp_path, _waiver_fixture())
    audit = tmp_path / P.HUMAN_AUDIT_REL
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(_audit_fixture(240, 240)))
    rel = P.reliability_gates(tmp_path)
    assert rel["status"] == P.GATE_PASS  # the REAL gates ran
    assert rel.get("waiver_ignored") is True


def test_waived_gate_is_nonblocking_and_verdict_carries_disclosure(tmp_path):
    _write_waiver(tmp_path, _waiver_fixture())
    cost = P.cost_report(tmp_path, tmp_path, "pilot", n_common=None)
    profiles = P.load_pilot_label_profile(tmp_path, "pilot")
    verdict = P.evaluate_gates(tmp_path, tmp_path, "pilot", profiles, None, None, cost)
    by_id = {g["gate_id"]: g for g in verdict["gates"]}
    assert by_id["human_audit_feasibility"]["status"] == P.GATE_WAIVED
    assert "human_audit_feasibility" not in verdict["blockers"]
    assert verdict["verdict"] == "PARK"  # everything else is still not-estimable
    assert verdict["waivers"] and verdict["waivers"][0]["plan_version"] == "v5"
    assert verdict["disclosures"] == [_waiver_fixture()["disclosure"]]
    # WAIVED is never PASS: the status string survives verbatim
    assert by_id["human_audit_feasibility"]["status"] != P.GATE_PASS


def _frame_manifest_fixture(tmp_path, empty_cell_key: str = "fact_questions|direct") -> Path:
    rows = []
    for row in C.ROW_IDS:
        per_cell = {}
        rf = P.F.FRAMES[row]
        for fr in rf.frames:
            for st in rf.strata:
                key = f"{fr.name}|{st.name}"
                per_cell[key] = (
                    [] if (row == "casualness" and key == empty_cell_key) else ["i0", "i1"]
                )
        rows.append({"row": row, "pilot_selection": {"per_cell_item_ids": per_cell}})
    p = tmp_path / "frame_manifest.json"
    p.write_text(json.dumps({"rows": rows}))
    return p


def _wave_summary_fixture(tmp_path, row: str, cells: list[str], calls: int = 100) -> Path:
    d = tmp_path / "judge" / "pilot" / row
    d.mkdir(parents=True, exist_ok=True)
    body = {
        "row": row,
        "split": "pilot",
        "dispatch_total_calls": calls,
        "counters": {"n_kept": calls, "n_kept_with_reasoning": calls, "n_api_refusal": 1},
        "not_estimable": {
            c: {"status": "not-estimable", "detail": "no frozen reference (synthetic)"}
            for c in cells
        },
    }
    p = d / "_wave_summary.json"
    p.write_text(json.dumps(body))
    return p


def test_load_declared_not_estimable_two_sources_and_malformed(tmp_path):
    fm = _frame_manifest_fixture(tmp_path)
    halluc_cells = P.expected_cells("hallucination")[:2]
    _wave_summary_fixture(tmp_path, "hallucination", halluc_cells)
    declared = P.load_declared_not_estimable(tmp_path, "pilot", frame_manifest_path=fm)
    assert set(declared) == set(halluc_cells) | {"casualness__fact_questions__direct"}
    for c in halluc_cells:
        assert declared[c]["source"] == "judge-wave-summary"
    assert (
        declared["casualness__fact_questions__direct"]["source"] == "frame-manifest-pilot-selection"
    )
    # malformed wave-summary record raises
    _wave_summary_fixture(tmp_path, "evil", [])
    ws = tmp_path / "judge" / "pilot" / "evil" / "_wave_summary.json"
    body = json.loads(ws.read_text())
    body["not_estimable"] = {P.expected_cells("evil")[0]: {"status": "wat"}}
    ws.write_text(json.dumps(body))
    with pytest.raises(P.PowerInputError, match="malformed"):
        P.load_declared_not_estimable(tmp_path, "pilot", frame_manifest_path=fm)
    ws.unlink()
    # foreign cell in the frame manifest raises
    fm_bad = json.loads(fm.read_text())
    fm_bad["rows"][0]["pilot_selection"]["per_cell_item_ids"]["nope|nah"] = []
    fm2 = tmp_path / "frame_manifest_bad.json"
    fm2.write_text(json.dumps(fm_bad))
    with pytest.raises(P.PowerInputError, match="foreign cell"):
        P.load_declared_not_estimable(tmp_path, "pilot", frame_manifest_path=fm2)


def test_profile_loader_declared_vs_undocumented(tmp_path):
    cell_declared = P.expected_cells(JUDGED_ROW)[0]
    declared = {
        cell_declared: {"source": "judge-wave-summary", "reason": "synthetic", "artifact": "x"}
    }
    profiles = P.load_pilot_label_profile(tmp_path, "pilot", [JUDGED_ROW], declared=declared)
    prof = profiles[JUDGED_ROW]
    assert cell_declared in prof.declared_not_estimable
    assert cell_declared not in prof.missing_cells
    assert len(prof.missing_cells) == 11  # the OTHER absences stay undocumented
    # a PRESENT artifact for a declared cell is a stale declaration => raises
    d = tmp_path / "judge" / "pilot" / JUDGED_ROW
    d.mkdir(parents=True)
    (d / f"{cell_declared}.json").write_text(json.dumps(_judge_cell_fixture(cell_declared, {})))
    with pytest.raises(P.PowerInputError, match="stale declaration"):
        P.load_pilot_label_profile(tmp_path, "pilot", [JUDGED_ROW], declared=declared)


def _full_profiles(declared_cells: dict[str, dict] | None = None, dead_rows=()) -> dict:
    """Synthetic profiles covering ALL 132 registered cells: estimable pools
    everywhere except dead rows (unanimous pools) and declared cells."""
    declared_cells = declared_cells or {}
    profiles = {}
    for row in C.ROW_IDS:
        prof = P.RowLabelProfile(row=row, judged=C.CONSTRUCTS[row].judge_scored)
        for cell in P.expected_cells(row):
            if cell in declared_cells:
                prof.declared_not_estimable[cell] = declared_cells[cell]
                continue
            if row in dead_rows:
                prof.cells[cell] = [(0, 10), (10, 10), (0, 10)]  # unanimous only
            else:
                prof.cells[cell] = [(3, 10), (5, 10), (2, 10)]
        profiles[row] = prof
    return profiles


def test_gate_discordance_declared_and_zero_disc_pass(tmp_path):
    declared = {
        P.expected_cells("hallucination")[0]: {
            "source": "judge-wave-summary",
            "reason": "no frozen reference (synthetic)",
            "artifact": "x",
        },
        "casualness__fact_questions__direct": {
            "source": "frame-manifest-pilot-selection",
            "reason": "zero eligible pilot prompts (synthetic)",
            "artifact": "y",
        },
    }
    profiles = _full_profiles(declared, dead_rows=("evil", "impoliteness"))
    disc = P.measure_discordance(profiles, seed=0)
    # declared cells carry source+reason in the record
    hall = disc["rows"]["hallucination"]["cells"][P.expected_cells("hallucination")[0]]
    assert hall["status"] == "declared-not-estimable"
    assert hall["source"] == "judge-wave-summary"
    gate = P._gate_discordance(disc, tmp_path)
    assert gate.status == P.GATE_PASS  # zero-disc + declared cells never FAIL it
    m = gate.measured
    assert m["n_registered"] == 132
    assert m["n_declared_not_estimable"] == 2
    assert m["n_zero_discordance_not_estimable"] == 24  # 2 dead rows x 12 cells
    assert m["n_estimable"] == 132 - 2 - 24
    assert "evil" not in m["rows_with_estimable_cells"]
    assert any("zero pilot discordance" in s for s in m["cells_not_estimable_zero_discordance"])
    # an UNDOCUMENTED absence still parks
    profiles["refusal"].missing_cells.append(profiles["refusal"].cells.popitem()[0])
    disc2 = P.measure_discordance(profiles, seed=0)
    gate2 = P._gate_discordance(disc2, tmp_path)
    assert gate2.status == P.GATE_NOT_ESTIMABLE and "UNDOCUMENTED" in gate2.detail


def test_verdict_carries_estimability_rollup(tmp_path):
    profiles = _full_profiles(dead_rows=("evil",))
    disc = P.measure_discordance(profiles, seed=0)
    cost = P.cost_report(tmp_path, tmp_path, "pilot", n_common=None)
    verdict = P.evaluate_gates(tmp_path, tmp_path, "pilot", profiles, disc, None, cost)
    assert verdict["rows_dead"] == ["evil"]
    assert len(verdict["cells_estimable"]) == 120
    assert len(verdict["cells_not_estimable"]) == 12
    assert all("zero pilot discordance" in v for v in verdict["cells_not_estimable"].values())


def test_select_production_n_dead_rows_not_simulated(tmp_path):
    reg = P.PowerRegistry(
        n_replicates=6,
        n_permutations=219,
        power_curve_effects=(0.70,),
        prompts_per_cell_floor=4,
        bisection_cap=16,
        primary_effect_auroc=0.70,
    )
    profiles = {
        r: p
        for r, p in _full_profiles(dead_rows=("evil",)).items()
        if r in ("evil", JUDGED_ROW, OBJECTIVE_ROW)
    }
    disc = P.measure_discordance(profiles, seed=0)
    ledger = P.PowerLedger(tmp_path / "units.jsonl")
    sel = P.select_production_n(profiles, disc, ledger, reg=reg, n_reps=6, n_perm=219, seed=0)
    assert sel["rows_dead"] == ["evil"]
    assert sorted(sel["rows_simulated"]) == sorted([JUDGED_ROW, OBJECTIVE_ROW])
    dead = sel["per_row_power_n"]["evil"]
    assert dead["n_power"] is None
    assert dead["status"] == "not-estimable: zero pilot discordance in every cell"
    assert sel["rows_missing_labels_undocumented"] == []
    # the dead row's unanimous cells are reported, never a veto
    assert sel["cells_not_estimable_zero_discordance"]
    if sel["status"] == "measured":
        assert sel["n_common"] >= reg.prompts_per_cell_floor
        # no ledger unit was burned on the dead row
        recs = [json.loads(x) for x in (tmp_path / "units.jsonl").read_text().strip().splitlines()]
        assert all(r["row"] != "evil" for r in recs)


def test_gate_power_coverage_is_simulated_plus_dead(tmp_path):
    sel = _full_coverage_selection()
    sel["rows_simulated"] = sorted(set(C.ROW_IDS) - {"evil", "casualness"})
    sel["rows_dead"] = ["casualness", "evil"]
    gate = P._gate_power(sel, tmp_path)
    assert gate.status == P.GATE_PASS
    assert gate.measured["rows_dead"] == ["casualness", "evil"]
    # dropping a row from BOTH sets fails coverage
    sel2 = dict(sel, rows_dead=["casualness"])
    gate2 = P._gate_power(sel2, tmp_path)
    assert gate2.status == P.GATE_FAIL and "evil" in gate2.detail
    # an undocumented missing-label row blocks even at full coverage
    sel3 = dict(sel, rows_missing_labels_undocumented=["refusal"])
    gate3 = P._gate_power(sel3, tmp_path)
    assert gate3.status == P.GATE_FAIL and "UNDOCUMENTED" in gate3.detail


def test_ledger_keys_stable_across_the_v5_profile_change():
    prof = P.RowLabelProfile(row=JUDGED_ROW, judged=True)
    cell = P.expected_cells(JUDGED_ROW)[0]
    prof.cells[cell] = [(3, 10), (1, 10), (5, 10)]
    base_fp = P.profile_fingerprint({JUDGED_ROW: prof})
    # pinned literal: the fingerprint MUST NOT change when RowLabelProfile
    # gains fields (declared_not_estimable) — resume keys cover cells ONLY
    assert base_fp == "e57ae1e6d1f758af4e9c04a41250176e70182f570b774de4796658fb38dfcef1"
    params = dict(
        row=JUDGED_ROW,
        n_prompts_per_cell=30,
        effect_auroc=0.6,
        alpha=P.REGISTERED.alpha_worst_case,
        n_reps=400,
        n_perm=659,
        responses_per_prompt=30,
        seed=0,
        profile_sha256=base_fp,
    )
    key_before = P.PowerLedger.unit_key(**params)
    prof.declared_not_estimable["x__y__z"] = {"source": "s", "reason": "r", "artifact": "a"}
    prof.missing_cells.append("q__w__e")
    assert P.profile_fingerprint({JUDGED_ROW: prof}) == base_fp
    assert P.PowerLedger.unit_key(**params) == key_before


def _write_gen_log(path: Path, cells: list[tuple[str, int, bool, int]], init_s: int = 60):
    lines = [
        "INFO 09-03 07:00:00 [core.py:1] engine start",
        f"INFO 09-03 07:{init_s // 60:02d}:{init_s % 60:02d} [gpu_model_runner.py:1] "
        "Graph capturing finished in 4 secs",
    ]
    for i, (cell, records, resumed, elapsed) in enumerate(cells, start=1):
        lines.append(
            f"[gen] cell {i}/{len(cells)} {cell} records={records} "
            f"resumed={resumed} elapsed={elapsed}s"
        )
    path.write_text("\n".join(lines) + "\n")


def test_timing_parser_on_synthetic_logs(tmp_path):
    import issue2658_pilot_timing as T

    logs = tmp_path / "logs"
    logs.mkdir()
    _write_gen_log(
        logs / "generate_shard00.log",
        [("a__b__c", 100, False, 10), ("d__e__f", 100, False, 25)],
        init_s=60,
    )
    _write_gen_log(
        logs / "generate_shard01.log",
        [("a__b__c", 100, True, 0), ("d__e__f", 100, False, 5)],
        init_s=90,
    )
    (logs / "capture_shard00.log").write_text(
        "[capture-shard00] rows 8/300 elapsed=1s\n"
        "[capture-shard00] rows 300/300 elapsed=10s\n"
        "[capture-shard01] rows 100/100 elapsed=5s\n"
    )
    (logs / "launcher_main.log").write_text(
        "[phase=p1_width] width=8 start\nnoise\n[phase=p1_width] width=8 restart\n"
    )
    (tmp_path / "gen_order_manifest").mkdir()
    (tmp_path / "gen_order_manifest" / "pilot_shard00of01.json").write_text(
        json.dumps({"n_requests": 400})
    )
    t = T.build_timing(logs, "2026-09-03T04:00:00Z", "2026-09-03T07:00:00Z", 8, tmp_path)
    assert t["wall_hours"] == 3.0 and t["gpu_hours_all_in"] == 24.0
    # shard01 has a resumed cell => EXCLUDED; rate from shard00 only: 25s/200
    assert t["shards_used_for_gen_rate"] == ["generate_shard00"]
    assert t["shards_excluded_resumed"] == ["generate_shard01"]
    assert abs(t["gen_marginal_s_per_response_per_gpu"] - 25 / 200) < 1e-12
    # engine init: 60s + 90s summed over the wave
    assert abs(t["fixed_overhead_hours"] - 150 / 3600) < 1e-12
    assert abs(t["capture_rows_per_s_per_gpu"] - 400 / 15) < 1e-9
    assert t["n_responses"] == 400
    assert t["capture_model_load_s_per_shard"]["basis"] == "not-measured"
    assert "2 [phase=p1_width] starts" in t["crash_fix_rounds_note"]
    # an incomplete capture sub-shard raises (never defaulted)
    (logs / "capture_shard00.log").write_text("[capture-shard00] rows 8/300 elapsed=1s\n")
    with pytest.raises(T.TimingParseError, match="incomplete"):
        T.build_timing(logs, "2026-09-03T04:00:00Z", "2026-09-03T07:00:00Z", 8, tmp_path)
    # a shard with no vLLM stamps raises
    (logs / "capture_shard00.log").write_text(
        "[capture-shard00] rows 300/300 elapsed=10s\n[capture-shard01] rows 100/100 elapsed=5s\n"
    )
    (logs / "generate_shard00.log").write_text(
        "[gen] cell 1/1 a__b__c records=1 resumed=False elapsed=1s\n"
    )
    with pytest.raises(T.TimingParseError, match="stamps"):
        T.build_timing(logs, "2026-09-03T04:00:00Z", "2026-09-03T07:00:00Z", 8, tmp_path)


def test_realized_judge_draws_summed_from_wave_summaries(tmp_path):
    halluc = P.expected_cells("hallucination")[:2]
    _wave_summary_fixture(tmp_path, "evil", [], calls=3000)
    _wave_summary_fixture(tmp_path, "hallucination", halluc, calls=1300)
    spend = {
        "schema": "i2658-judge-spend-v1",
        "dollars": 43.0,
        "basis": "priced from measured tokens, not billed",
        "price_source_url": "https://example.test/pricing",
        "rates_per_mtok": {"input_per_mtok": 1.5, "output_per_mtok": 7.5},
        "n_calls_succeeded": 4300,
        "per_call_mean_input_tokens": 2000.0,
        "per_call_mean_output_tokens": 300.0,
    }
    sp = tmp_path / P.JUDGE_SPEND_REL
    sp.parent.mkdir(parents=True)
    sp.write_text(json.dumps(spend))
    rep = P.cost_report(tmp_path, tmp_path, "pilot", n_common=48)
    draws = rep["api"]["realized_judge_draws"]
    assert draws["value"] == 4300  # dispatch_total_calls, NOT the counters sum
    assert "dispatch_total_calls" in draws["detail"]
    calls = rep["api"]["projected_production_judge_calls"]
    # 8 judged rows x 12 cells - 2 declared no-reference = 94 judgeable cells
    assert calls["value"] == 94 * 48 * 30 * 5 * 2
    dollars = rep["api"]["projected_production_judge_dollars"]
    assert abs(dollars["value"] - calls["value"] * 43.0 / 4300) < 1e-9
    assert dollars["basis"] == "projected"
    assert rep["api"]["measured_dollars"]["value"] == 43.0
    assert rep["api"]["measured_dollars"]["basis"] == "priced from measured tokens, not billed"


def test_stager_writes_provenance_offline(tmp_path, monkeypatch):
    import huggingface_hub as hf

    import explore_persona_space.orchestrate.hub as hub_mod

    prefix = f"{P.G.EXPERIMENT_NAME}/analysis_tensors/l19_pilot"
    files = {}
    for i in range(8):
        shard = f"shard{i:02d}of08"
        for name in (
            "row_index_shard00.jsonl",
            "row_index_shard01.jsonl",
            "_capture_manifest.json",
            "_capture_meta_shard00.json",
            "_capture_meta_shard01.json",
            "l19mean_shard00.npy",
        ):
            # row_index shards stay EMPTY (valid JSONL) so the alignment
            # gate can parse them; the manifest carries a payload marker
            files[f"{prefix}/{shard}/{name}"] = (
                "" if name.startswith("row_index") else f"payload:{shard}/{name}\n"
            )

    class _FakeApi:
        def repo_info(self, repo, repo_type):
            assert repo_type == "dataset"
            return type("RI", (), {"sha": "f" * 40})()

    def _fake_list(api, repo, *, repo_type, revision, path_in_repo):
        assert revision == "f" * 40 and path_in_repo == prefix
        return sorted((p, len(v)) for p, v in files.items())

    def _fake_download(*, repo_id, filename, repo_type, revision):
        assert revision == "f" * 40
        local = tmp_path / "hfcache" / filename.replace("/", "_")
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(files[filename])
        return str(local)

    monkeypatch.setattr(hf, "HfApi", _FakeApi)
    monkeypatch.setattr(hf, "hf_hub_download", _fake_download)
    monkeypatch.setattr(hub_mod, "list_repo_entries_complete", _fake_list)
    dest = P.stage_store_index_from_hub(tmp_path / "out", "pilot")
    prov = json.loads((dest / "_staged_from_hub.json").read_text())
    assert prov["revision"] == "f" * 40 and prov["prefix"] == prefix
    assert prov["n_files"] == 8 * 5  # sidecars only, NEVER the .npy
    assert all(not f["path_in_repo"].endswith(".npy") for f in prov["files"])
    assert (dest / "shard03of08" / "_capture_manifest.json").read_text().startswith("payload:")
    # the alignment gate reports the staged provenance (empty store + empty
    # gen dir align vacuously; the store_source string is what is under test).
    # r12/g1 concern 3: point the gate at a MINIMAL provenance fixture in the
    # tmp root so this test never reads the committed direction_provenance.json.
    gen_dir = tmp_path / "out" / "raw_completions" / "pilot"
    gen_dir.mkdir(parents=True)
    prov_path = tmp_path / "out" / "direction_provenance.json"
    prov_path.write_text(
        json.dumps({"rows": [{"row": "evil", "c2_c3": "eligible", "shape": [3584]}]})
    )
    monkeypatch.setattr(P.F, "PROVENANCE_PATH", prov_path)
    gate = P._gate_row_vector_alignment(tmp_path / "out", tmp_path / "out", "pilot")
    assert "hub-staged (" + "f" * 40 + ")" in str(gate.measured)


def _write_gen_summaries(
    tmp_path,
    split: str,
    frac_by_cell: dict[str, float],
    n_shards: int = 2,
    n_per_cell: int = 50,
    threshold_override: float | None = None,
) -> list:
    """Multi-shard gen summaries whose cap_hit blocks are DERIVED via the real
    ``G.cap_hit_report`` over synthetic finish_reason rows (r12/g2 nit 5:
    never a re-implemented rule), in the observed unit-5 schema."""
    d = tmp_path / "gen_summary"
    d.mkdir(parents=True, exist_ok=True)
    keys = sorted(frac_by_cell)
    paths = []
    for si in range(n_shards):
        rows = []
        for key in keys[si::n_shards]:
            row, frame, band = key.split("|")
            n_hit = round(frac_by_cell[key] * n_per_cell)
            rows.extend(
                {
                    "row": row,
                    "cell": f"{frame}|{band}",
                    "finish_reason": "length" if j < n_hit else "stop",
                }
                for j in range(n_per_cell)
            )
        rep = P.G.cap_hit_report(rows)
        if threshold_override is not None:
            rep["threshold"] = threshold_override
        body = {"split": split, "shard": f"shard{si:02d}of{n_shards:02d}", "cap_hit": rep}
        path = d / f"{split}_shard{si:02d}of{n_shards:02d}.json"
        path.write_text(json.dumps(body))
        paths.append(path)
    return paths


def _all_cell_keys(frac: float = 0.0) -> dict[str, float]:
    return {cell.replace("__", "|"): frac for row in C.ROW_IDS for cell in P.expected_cells(row)}


def _raw_decoder_fixture(tmp_path, split: str = "pilot", cap: int = 1024) -> None:
    d = tmp_path / "raw_completions" / split
    d.mkdir(parents=True, exist_ok=True)
    for cell in ("evil__arc_c_tasks__direct", "evil__arc_c_tasks__indirect"):
        (d / f"{cell}.json").write_text(
            json.dumps({"schema": "i2658-gen-cell-v1", "decoder": {"max_new_tokens": cap}})
        )


def test_gate_cap_hit_reads_shard_summaries(tmp_path):
    all_keys = _all_cell_keys()
    _write_gen_summaries(tmp_path, "pilot", all_keys)
    gate = P._gate_cap_hit(tmp_path, tmp_path, "pilot")
    assert gate.status == P.GATE_PASS
    assert gate.measured["n_cells_covered"] == 132
    assert gate.measured["n_shard_summaries"] == 2
    # one cell over the 2% threshold with NO amendment record => honest FAIL
    hot = dict(all_keys)
    hot_key = sorted(hot)[0]
    hot[hot_key] = 0.16
    _write_gen_summaries(tmp_path, "pilot", hot)
    gate2 = P._gate_cap_hit(tmp_path, tmp_path, "pilot")
    assert gate2.status == P.GATE_FAIL and "no amendment record" in gate2.detail
    assert gate2.measured["worst_per_cell_fraction"] == 0.16
    assert hot_key.replace("|", "__") in gate2.detail
    # incomplete coverage without a declaration stays NOT-ESTIMABLE, and the
    # r12/g2 concern-2 detail NAMES the missing cells
    partial = dict(sorted(all_keys.items())[:100])
    _write_gen_summaries(tmp_path, "pilot", partial)
    gate3 = P._gate_cap_hit(tmp_path, tmp_path, "pilot")
    assert gate3.status == P.GATE_NOT_ESTIMABLE
    a_missing = sorted(set(all_keys) - set(partial))[0].replace("|", "__")
    assert a_missing in gate3.detail
    # a declared never-generated cell shrinks the denominator instead
    missing_cell = sorted(set(all_keys) - set(partial))[0].replace("|", "__")
    declared = {
        missing_cell: {
            "source": "frame-manifest-pilot-selection",
            "reason": "zero eligible pilot prompts (synthetic)",
            "artifact": "y",
        }
    }
    almost = {k: v for k, v in all_keys.items() if k.replace("|", "__") != missing_cell}
    _write_gen_summaries(tmp_path, "pilot", almost)
    gate4 = P._gate_cap_hit(tmp_path, tmp_path, "pilot", declared)
    assert gate4.status == P.GATE_PASS and gate4.measured["n_expected"] == 131


def test_gate_cap_hit_input_errors(tmp_path):
    # threshold drift in a shard summary raises, naming the shard path (g2 c1)
    a = tmp_path / "a"
    hot = _all_cell_keys()
    hot[sorted(hot)[0]] = 0.16
    _write_gen_summaries(a, "pilot", hot, threshold_override=0.05)
    with pytest.raises(P.PowerInputError, match="shard00of02"):
        P._gate_cap_hit(a, a, "pilot")
    # a cell key outside the registered grid raises, naming it (g2 c2)
    b = tmp_path / "b"
    foreign = _all_cell_keys()
    foreign["evil|not_a_frame|direct"] = 0.0
    _write_gen_summaries(b, "pilot", foreign)
    with pytest.raises(P.PowerInputError, match="not_a_frame"):
        P._gate_cap_hit(b, b, "pilot")
    # a malformed shard summary raises with the path, never a bare KeyError (g2 nit 3)
    c = tmp_path / "c"
    d = c / "gen_summary"
    d.mkdir(parents=True)
    (d / "pilot_shard00of01.json").write_text(json.dumps({"split": "pilot"}))
    with pytest.raises(P.PowerInputError, match="malformed"):
        P._gate_cap_hit(c, c, "pilot")


def test_cap_amendment_producer_on_synthetic_shards(tmp_path):
    keys = _all_cell_keys()
    offender_keys = sorted(keys)[:3]
    fracs = (0.16, 0.58, 0.04)
    for k, f in zip(offender_keys, fracs, strict=True):
        keys[k] = f
    _write_gen_summaries(tmp_path, "pilot", keys)
    _raw_decoder_fixture(tmp_path)
    rec = P.build_cap_amendment(tmp_path, tmp_path, "pilot")
    assert rec["schema"] == P.CAP_AMENDMENT_SCHEMA and rec["plan_version"] == "v6"
    assert rec["pilot_max_new_tokens"] == 1024  # derived from the decoder records
    assert rec["production_max_new_tokens"] == C.PRODUCTION_MAX_NEW_TOKENS == 4096
    assert sorted(rec["cells_over_threshold"]) == sorted(offender_keys)
    assert rec["n_offender_cells"] == 3
    assert rec["n_truncated_records"] == sum(round(f * 50) for f in fracs)
    assert rec["n_records_total"] == 132 * 50
    assert rec["threshold"] == P.G.CAP_HIT_AMEND_THRESHOLD
    assert rec["registered_rule_plan_v4_section5"].startswith("realized length-cap fraction")
    assert "truncated answers" in rec["disclosure"]
    assert (tmp_path / P.CAP_AMENDMENT_REL).exists()
    for k in offender_keys:
        assert rec["cells_over_threshold"][k]["n"] == 50


def test_cap_amendment_producer_fails_loud(tmp_path):
    keys = _all_cell_keys()
    _raw_decoder_fixture(tmp_path)
    # an EMPTY offender set is a wiring error, not a record
    _write_gen_summaries(tmp_path, "pilot", keys)
    with pytest.raises(P.PowerInputError, match="ZERO cells"):
        P.build_cap_amendment(tmp_path, tmp_path, "pilot")
    # a missing shard summary fails loud (shardXXofYY completeness)
    hot = dict(keys)
    hot[sorted(hot)[0]] = 0.16
    paths = _write_gen_summaries(tmp_path, "pilot", hot)
    paths[1].unlink()
    with pytest.raises(P.PowerInputError, match="missing pilot shard summaries"):
        P.build_cap_amendment(tmp_path, tmp_path, "pilot")
    # mixed decoder caps across raw-completion bodies fail loud
    _write_gen_summaries(tmp_path, "pilot", hot)
    (tmp_path / "raw_completions" / "pilot" / "z.json").write_text(
        json.dumps({"decoder": {"max_new_tokens": 2048}})
    )
    with pytest.raises(P.PowerInputError, match="mixed decoder caps"):
        P.build_cap_amendment(tmp_path, tmp_path, "pilot")


def test_gate_cap_hit_amended_and_fail_paths(tmp_path):
    keys = _all_cell_keys()
    offenders = sorted(keys)[:12]  # 12 > 10: r12/g2 nit 4, detail never truncated
    for k in offenders:
        keys[k] = 0.16
    _write_gen_summaries(tmp_path, "pilot", keys)
    _raw_decoder_fixture(tmp_path)
    rec = P.build_cap_amendment(tmp_path, tmp_path, "pilot")
    gate = P._gate_cap_hit(tmp_path, tmp_path, "pilot")
    assert gate.status == P.GATE_AMENDED
    assert gate.status != P.GATE_PASS  # AMENDED is never reported PASS
    for k in offenders:
        assert k.replace("|", "__") in gate.detail
    assert gate.measured["amendment_record"]["production_max_new_tokens"] == 4096
    rec_path = tmp_path / P.CAP_AMENDMENT_REL
    # a record whose cap sits below the registered 2x floor keeps FAIL
    bad = dict(rec)
    bad["production_max_new_tokens"] = 1500
    rec_path.write_text(json.dumps(bad))
    gate2 = P._gate_cap_hit(tmp_path, tmp_path, "pilot")
    assert gate2.status == P.GATE_FAIL and "2x pilot cap" in gate2.detail
    # a record not covering one realized offender keeps FAIL, naming the cell
    bad2 = dict(rec)
    bad2["cells_over_threshold"] = {
        k: v for k, v in rec["cells_over_threshold"].items() if k != offenders[0]
    }
    rec_path.write_text(json.dumps(bad2))
    gate3 = P._gate_cap_hit(tmp_path, tmp_path, "pilot")
    assert gate3.status == P.GATE_FAIL and "not covered" in gate3.detail
    assert offenders[0].replace("|", "__") in gate3.detail


def test_amended_gate_is_nonblocking_and_verdict_carries_amendment(tmp_path):
    keys = _all_cell_keys()
    keys[sorted(keys)[0]] = 0.16
    _write_gen_summaries(tmp_path, "pilot", keys)
    _raw_decoder_fixture(tmp_path)
    P.build_cap_amendment(tmp_path, tmp_path, "pilot")
    cost = P.cost_report(tmp_path, tmp_path, "pilot", n_common=None)
    profiles = P.load_pilot_label_profile(tmp_path, "pilot")
    verdict = P.evaluate_gates(tmp_path, tmp_path, "pilot", profiles, None, None, cost)
    by_id = {g["gate_id"]: g for g in verdict["gates"]}
    assert by_id["measured_cap_hit_rate"]["status"] == P.GATE_AMENDED
    assert "measured_cap_hit_rate" not in verdict["blockers"]
    assert verdict["verdict"] == "PARK"  # everything else is still not-estimable
    assert verdict["amendments"] and verdict["amendments"][0]["production_max_new_tokens"] == 4096
    assert verdict["amendments"][0]["n_offender_cells"] == 1
    assert P.CAP_AMENDMENT_DISCLOSURE_V6A7 in verdict["disclosures"]


def test_cost_report_scales_gen_projection_under_cap_amendment(tmp_path):
    timing = tmp_path / P.PILOT_TIMING_REL
    timing.parent.mkdir(parents=True)
    timing.write_text(json.dumps(_timing_fixture()))
    keys = _all_cell_keys()
    keys[sorted(keys)[0]] = 0.16  # 8 truncated of 132 x 50 = 6600 records
    _write_gen_summaries(tmp_path, "pilot", keys)
    _raw_decoder_fixture(tmp_path)
    P.build_cap_amendment(tmp_path, tmp_path, "pilot")
    rep = P.cost_report(tmp_path, tmp_path, "pilot", n_common=30)
    proj = rep["gpu_hours"]["projected_production_gpu_h_measured_marginal"]
    scaling = proj["cap_amendment_scaling"]
    assert scaling["applied"] is True
    mult = 1.0 + (8 / 6600) * (4096 / 1024 - 1.0)
    assert abs(scaling["gen_marginal_multiplier"] - mult) < 1e-12
    n_prod = 11 * 12 * 30 * 30 * 2
    gen = n_prod * 0.14 * mult / 3600 + 2 * 0.135
    cap = n_prod / 35.0 / 3600
    assert abs(proj["value"] - (gen + cap)) < 1e-9
    assert "upper bound" in scaling["assumption"]
    # absent record: the projection stays unscaled and says so
    bare = tmp_path / "bare"
    timing2 = bare / P.PILOT_TIMING_REL
    timing2.parent.mkdir(parents=True)
    timing2.write_text(json.dumps(_timing_fixture()))
    rep2 = P.cost_report(bare, bare, "pilot", n_common=30)
    proj2 = rep2["gpu_hours"]["projected_production_gpu_h_measured_marginal"]
    assert proj2["cap_amendment_scaling"]["applied"] is False
    assert abs(proj2["value"] - (n_prod * 0.14 / 3600 + 2 * 0.135 + cap)) < 1e-9


def test_resolve_max_new_tokens_split_behavior(tmp_path):
    G = P.G
    assert G.resolve_max_new_tokens("pilot", eval_root=tmp_path) == 1024
    with pytest.raises(G.GenerationBudgetError, match="cap amendment record"):
        G.resolve_max_new_tokens("dev", eval_root=tmp_path)
    with pytest.raises(ValueError, match="unknown split"):
        G.resolve_max_new_tokens("prod", eval_root=tmp_path)
    rec_path = tmp_path / G.CAP_AMENDMENT_REL
    rec_path.parent.mkdir(parents=True)
    rec_path.write_text(
        json.dumps(
            {
                "schema": P.CAP_AMENDMENT_SCHEMA,
                "plan_version": "v6",
                "pilot_max_new_tokens": 1024,
                "production_max_new_tokens": 4096,
            }
        )
    )
    assert G.resolve_max_new_tokens("dev", eval_root=tmp_path) == 4096
    assert G.resolve_max_new_tokens("test", eval_root=tmp_path) == 4096
    # pilot stays 1024 even with the record present
    assert G.resolve_max_new_tokens("pilot", eval_root=tmp_path) == 1024
    assert G.prompt_budget_for_cap(4096) == G.MAX_MODEL_LEN - 4096 == 4096
    assert G.prompt_budget_for_cap(1024) == 7168
    with pytest.raises(G.GenerationBudgetError, match="no prompt budget"):
        G.prompt_budget_for_cap(G.MAX_MODEL_LEN)
    rec_path.write_text(json.dumps({"schema": "x"}))
    with pytest.raises(G.GenerationBudgetError, match="missing field"):
        G.resolve_max_new_tokens("dev", eval_root=tmp_path)


def test_generation_fingerprint_carries_realized_cap():
    G = P.G
    cw = G.CellWork(
        row="evil",
        frame="arc_c_tasks",
        band="direct",
        item_ids=("i0",),
        superfamilies={"i0": "s0"},
    )
    fp_pilot_1024 = G.generation_fingerprint(cw, 10, "pilot", 1024)
    assert fp_pilot_1024 != G.generation_fingerprint(cw, 10, "pilot", 4096)
    assert fp_pilot_1024 != G.generation_fingerprint(cw, 10, "dev", 4096)
    # deterministic for a fixed cap (the pilot resume contract)
    assert fp_pilot_1024 == G.generation_fingerprint(cw, 10, "pilot", 1024)
