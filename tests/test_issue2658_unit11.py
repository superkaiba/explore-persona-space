"""Issue #2658 unit-11 tests: preregistered confirmatory inference (plan §7).

Pins (each guard shown FIRING, not merely absent):
- estimator-parity link: the inference module's primitives are unit 8's
  registered functions BY IDENTITY, and the permutation battery actually
  calls them;
- the within-exact-prompt permutation preserves each prompt's label multiset
  and never crosses prompts;
- plus-one p-value arithmetic (p == 0 is impossible; out-of-range raises);
- the Monte Carlo family-wide extension trigger, and that extension is
  FAMILY-WIDE (every test extends, initial chunks reused, never redrawn);
- Holm adjustment on a hand-computed case, incl. the realized family sizes
  10/11/10 derived from the COMMITTED partition artifact;
- one-sidedness of the C5-minus-C2 studentized bootstrap;
- the PRE-REGISTERED bootstrap-family deterministic extension (2,000 ->
  20,000): family-wide on a single-row overlap, initial chunks reused (their
  persisted draws summed into the extended p, never redrawn), the unfired
  verdict recorded, and the record naming itself an ADDITION to plan §7;
- pooled-fold consumption RAISES (unit 9's structural dead end);
- a missing prospective ledger RAISES (never "all cells estimable");
- the ledger-driven per-row denominator revision (excluded cells dropped,
  causes + revised denominators recorded);
- row-level production-gate failure returns not-estimable (no proxy);
- checkpoint/resume on machine-stable generating-parameter keys, and the
  obs-stat drift guard on a tampered ledger record.

All tests are OFFLINE and synthetic: no GPU, no network, no judge API call,
no bank item text. The committed-partition pin follows the sibling
convention (skip in a checkout without the built artifact).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2658_common as C  # noqa: E402
import issue2658_comparators as U  # noqa: E402
import issue2658_inference as INF  # noqa: E402
import issue2658_power as PW  # noqa: E402

TINY_REG = INF.InferenceRegistry(
    n_perm_initial=40,
    perm_chunk_initial=(20, 20),
    n_perm_extended=100,
    perm_chunk_extension=(30, 30),
    n_boot=40,
    boot_chunk=20,
    n_ci_draws=50,
    min_discordant_prompts=5,
    min_answers_per_class=10,
    min_prompts_per_class=2,
)


def _panel(row: str, *, n_prompts=12, m=6, effect=1.5, seed=0, comps=("c2_direction_dot",)):
    rng = np.random.default_rng(seed)
    n = n_prompts * m
    pids = np.repeat([f"{row}-p{i:03d}" for i in range(n_prompts)], m)
    labels = rng.random(n) < 0.5
    scores = {c: rng.standard_normal(n) + effect * labels for c in comps}
    return INF.InferencePanel(
        row=row,
        prompt_ids=pids,
        response_index=np.tile(np.arange(m, dtype=np.int64), n_prompts),
        labels=labels,
        scores=scores,
        cells=np.array(["f|b"] * n),
        accounting={"excluded_cells": [], "cells_total": 1, "cells_used": 1, "floor": 15},
    )


# ---------------------------------------------------------------------------
# Estimator-parity link (identity, then actually-called).
# ---------------------------------------------------------------------------
def test_registered_primitives_are_unit8_functions_by_identity():
    assert INF.MACRO_AUROC is PW.equal_prompt_macro_auroc
    assert INF.PERMUTE_WITHIN_PROMPT is PW.permute_labels_within_prompt
    assert INF.WITHIN_PROMPT_AUROC is PW.within_prompt_auroc


def test_permutation_battery_calls_the_registered_primitives(tmp_path, monkeypatch):
    calls = {"perm": 0, "macro": 0}
    real_perm, real_macro = PW.permute_labels_within_prompt, PW.equal_prompt_macro_auroc

    def perm_spy(*a, **k):
        calls["perm"] += 1
        return real_perm(*a, **k)

    def macro_spy(*a, **k):
        calls["macro"] += 1
        return real_macro(*a, **k)

    monkeypatch.setattr(INF, "PERMUTE_WITHIN_PROMPT", perm_spy)
    monkeypatch.setattr(INF, "MACRO_AUROC", macro_spy)
    panel = _panel("rowA")
    led = INF.InfLedger(tmp_path / "perm.jsonl", INF.PERM_CHUNK_SCHEMA)
    res = INF.run_permutation_test(
        panel,
        "c2_direction_dot",
        scores_fingerprint="t",
        ledger=led,
        chunk_plan=TINY_REG.perm_chunk_initial,
    )
    assert calls["perm"] >= 2 and calls["macro"] >= 3  # obs + per sub-chunk
    assert res["n_perm"] == TINY_REG.n_perm_initial


# ---------------------------------------------------------------------------
# Within-exact-prompt permutation: multiset preserved, never crosses prompts.
# ---------------------------------------------------------------------------
def test_within_prompt_permutation_preserves_multiset_and_never_crosses():
    rng = np.random.default_rng(7)
    pids = np.array(["a"] * 4 + ["b"] * 6 + ["c"] * 3)
    labels = np.zeros(13, dtype=bool)
    labels[:3] = True  # all positives live in prompt 'a'
    perms = INF.PERMUTE_WITHIN_PROMPT(labels, pids, rng, n_perm=64)
    assert perms.shape == (64, 13)
    for pid, sl in (("a", slice(0, 4)), ("b", slice(4, 10)), ("c", slice(10, 13))):
        want = np.sort(labels[sl])
        got = np.sort(perms[:, sl], axis=1)
        assert (got == want).all(), f"prompt {pid} label multiset changed"
    # never crosses: prompts b/c never gain a positive.
    assert perms[:, 4:].sum() == 0


# ---------------------------------------------------------------------------
# Plus-one p arithmetic.
# ---------------------------------------------------------------------------
def test_plus_one_p_is_never_zero_and_validates_range():
    assert INF.plus_one_p(0, 9999) == pytest.approx(1.0 / 10000.0)
    assert INF.plus_one_p(0, 9999) > 0.0
    assert INF.plus_one_p(9999, 9999) == 1.0
    with pytest.raises(INF.InferenceInputError):
        INF.plus_one_p(-1, 9999)
    with pytest.raises(INF.InferenceInputError):
        INF.plus_one_p(10000, 9999)


# ---------------------------------------------------------------------------
# Family-wide deterministic extension.
# ---------------------------------------------------------------------------
def test_extension_trigger_pure_function():
    th = (0.005, 0.05)
    hit = INF.extension_trigger({"a": (0.001, 0.01), "b": (0.4, 0.6)}, th)
    assert hit["triggered"] and hit["overlapping_tests"] == {"a": [0.005]}
    miss = INF.extension_trigger({"a": (0.06, 0.2), "b": (0.4, 0.6)}, th)
    assert not miss["triggered"] and miss["overlapping_tests"] == {}


def test_extension_is_family_wide_and_reuses_initial_chunks(tmp_path, monkeypatch):
    # Force the trigger with only ONE overlapping test: EVERY test must extend.
    monkeypatch.setattr(
        INF,
        "extension_trigger",
        lambda mc, th: {
            "triggered": True,
            "overlapping_tests": {"rowA": [0.05]},
            "thresholds": list(th),
        },
    )
    panels = {
        "rowA": (_panel("rowA", seed=1), "shaA"),
        "rowB": (_panel("rowB", seed=2), "shaB"),
    }
    led = INF.InfLedger(tmp_path / "perm.jsonl", INF.PERM_CHUNK_SCHEMA)
    fam = INF.run_family_permutations("C2", "c2_direction_dot", panels, 2, TINY_REG, led)
    # Family-wide: BOTH rows at the extended draw count, not just the
    # borderline one.
    assert fam["n_perm_realized"] == {
        "rowA": TINY_REG.n_perm_extended,
        "rowB": TINY_REG.n_perm_extended,
    }
    recs = [json.loads(x) for x in (tmp_path / "perm.jsonl").read_text().splitlines()]
    # Initial chunks were REUSED, never redrawn: exactly one record per
    # (row, chunk_start) across both passes.
    starts = sorted((r["row"], r["chunk_start"]) for r in recs)
    assert starts == sorted((row, s) for row in ("rowA", "rowB") for s in (0, 20, 40, 70))
    # Determinism of a seeded chunk: recomputing chunk 0 in a fresh ledger
    # yields the identical exceedance count.
    led2 = INF.InfLedger(tmp_path / "perm2.jsonl", INF.PERM_CHUNK_SCHEMA)
    INF.run_permutation_test(
        panels["rowA"][0],
        "c2_direction_dot",
        scores_fingerprint="shaA",
        ledger=led2,
        chunk_plan=(20,),
    )
    rec0 = next(r for r in recs if r["row"] == "rowA" and r["chunk_start"] == 0)
    rec0b = json.loads((tmp_path / "perm2.jsonl").read_text().splitlines()[0])
    assert rec0b["exceed"] == rec0["exceed"] and rec0b["seed"] == rec0["seed"]


# ---------------------------------------------------------------------------
# Holm: hand-computed case + realized family sizes from the committed artifact.
# ---------------------------------------------------------------------------
def test_holm_adjust_hand_computed_at_m10():
    p = {"a": 0.001, "b": 0.004, "c": 0.03}
    adj = INF.holm_adjust(p, m=10)
    assert adj["a"] == pytest.approx(0.01)  # 10 * .001
    assert adj["b"] == pytest.approx(0.036)  # max(.01, 9 * .004)
    assert adj["c"] == pytest.approx(0.24)  # max(.036, 8 * .03)
    sig = {k: v <= 0.05 for k, v in adj.items()}
    assert sig == {"a": True, "b": True, "c": False}
    with pytest.raises(INF.InferenceInputError):
        INF.holm_adjust({"a": 0.1, "b": 0.2}, m=1)  # family smaller than tests
    with pytest.raises(INF.InferenceInputError):
        INF.holm_adjust({"a": 0.0}, m=3)  # p == 0 impossible under plus-one


def test_realized_family_sizes_derive_from_committed_partition():
    prov_path = REPO_ROOT / "eval_results/issue_2658/direction_provenance.json"
    if not prov_path.exists():
        pytest.skip("direction_provenance.json not built in this checkout")
    prov = U.load_committed_provenance(prov_path)
    part = U.c2c3_partition(prov)
    sizes = INF.derive_family_sizes(part, C.ROW_IDS)
    assert sizes == {"C2": 10, "C5": 11, "C5_minus_C2": 10}
    assert sizes == C.holm_family_sizes(len(part["not_estimable"]))
    committed = prov["c2_c3_partition"].get("holm_family_sizes")
    if committed is not None:
        assert sizes == committed
    # The power basis 0.05/11 is a sizing worst case, NOT a family size.
    assert PW.REGISTERED.holm_family_size == 11


# ---------------------------------------------------------------------------
# One-sidedness of the C5-minus-C2 studentized bootstrap.
# ---------------------------------------------------------------------------
def _boot_setup(*, c5_signal: float, c2_signal: float, seed: int, row: str = "synthetic"):
    rd = U.synthesize_row_data(
        row=row, n_prompts=48, n_responses=6, d=8, n_superfamilies=12, effect=2.5, seed=seed
    )
    te = [i for i, r in enumerate(rd.rows) if r.split == "test"]
    rows = [rd.rows[i] for i in te]
    x = rd.X[te].astype(np.float64)
    w = np.asarray(rd.synthetic_direction, dtype=np.float64)
    rng = np.random.default_rng(seed + 1)
    sig = x @ w
    panel = INF.InferencePanel(
        row=row,
        prompt_ids=np.array([r.prompt_id for r in rows]),
        response_index=np.array([r.response_index for r in rows], dtype=np.int64),
        labels=np.array([r.label for r in rows], dtype=bool),
        scores={
            "c5_full_probe": c5_signal * sig + 0.5 * rng.standard_normal(len(te)),
            "c2_direction_dot": c2_signal * sig + 0.5 * rng.standard_normal(len(te)),
        },
        cells=np.array([f"{r.source_frame}|{r.stratum}" for r in rows]),
        accounting={"excluded_cells": [], "cells_total": 4, "cells_used": 4, "floor": 15},
    )
    return rd, panel


def test_bootstrap_is_one_sided_greater(tmp_path):
    rd_f, panel_f = _boot_setup(c5_signal=2.0, c2_signal=0.0, seed=3)
    res_f = INF.run_bootstrap_test(
        panel_f,
        rd_f,
        selected_c=1.0,
        scores_fingerprint="fwd",
        ledger=INF.InfLedger(tmp_path / "bf.jsonl", INF.BOOT_CHUNK_SCHEMA),
        reg=TINY_REG,
    )
    rd_r, panel_r = _boot_setup(c5_signal=0.0, c2_signal=2.0, seed=3)
    res_r = INF.run_bootstrap_test(
        panel_r,
        rd_r,
        selected_c=1.0,
        scores_fingerprint="rev",
        ledger=INF.InfLedger(tmp_path / "br.jsonl", INF.BOOT_CHUNK_SCHEMA),
        reg=TINY_REG,
    )
    assert res_f["delta_hat"] > 0 > res_r["delta_hat"]
    assert res_f["p"] < res_r["p"]  # H1: C5 > C2 — reversed effect must NOT reject
    assert res_r["p"] > 0.5
    assert res_f["sidedness"] == "greater"
    with pytest.raises(INF.InferenceInputError):
        INF.run_bootstrap_test(  # off-grid frozen C refuses
            panel_f,
            rd_f,
            selected_c=0.123,
            scores_fingerprint="x",
            ledger=INF.InfLedger(tmp_path / "bx.jsonl", INF.BOOT_CHUNK_SCHEMA),
            reg=TINY_REG,
        )


BOOT_EXT_REG = INF.InferenceRegistry(
    n_perm_initial=40,
    perm_chunk_initial=(20, 20),
    n_perm_extended=100,
    perm_chunk_extension=(30, 30),
    n_boot=40,
    boot_chunk=20,
    n_boot_extended=80,
    n_ci_draws=50,
    min_discordant_prompts=5,
    min_answers_per_class=10,
    min_prompts_per_class=2,
)


def _two_row_boot_family(tmp_path):
    rd_a, panel_a = _boot_setup(c5_signal=1.0, c2_signal=0.5, seed=5, row="rowA")
    rd_b, panel_b = _boot_setup(c5_signal=0.5, c2_signal=1.0, seed=6, row="rowB")
    return {
        "rowA": (panel_a, rd_a, 1.0, "shaA"),
        "rowB": (panel_b, rd_b, 1.0, "shaB"),
    }


def test_bootstrap_extension_is_family_wide_and_reuses_chunks(tmp_path, monkeypatch):
    # Mocked trigger names ONE overlapping row: EVERY row must still extend.
    monkeypatch.setattr(
        INF,
        "extension_trigger",
        lambda mc, th: {
            "triggered": True,
            "overlapping_tests": {"rowA": [0.025]},
            "thresholds": list(th),
        },
    )
    led_path = tmp_path / "boot.jsonl"
    fam = INF.run_family_bootstrap(
        _two_row_boot_family(tmp_path),
        2,
        BOOT_EXT_REG,
        INF.InfLedger(led_path, INF.BOOT_CHUNK_SCHEMA),
    )
    ext = fam["extension"]
    assert ext["registered"] is True and ext["fired"] is True
    assert ext["n_boot_realized"] == {"rowA": 80, "rowB": 80}  # family-wide, not per-test
    assert all(res["n_boot"] == 80 for res in fam["tests"].values())
    # The record names itself an ADDITION to plan section 7, never plan-registered.
    assert "ADDITION to plan section 7" in ext["provenance"]
    assert ext["n_boot_initial"] == 40 and ext["n_boot_extended"] == 80
    recs = [json.loads(x) for x in led_path.read_text().splitlines()]
    starts = sorted((r["row"], r["chunk_start"]) for r in recs)
    # Initial chunks (0, 20) were REUSED from the ledger — exactly one record
    # per (row, chunk_start) across both passes, never redrawn.
    assert starts == sorted((row, s) for row in ("rowA", "rowB") for s in (0, 20, 40, 60))
    # The extended p sums the PERSISTED initial-chunk draws into the 80-draw
    # exceedance count (chunk records in start order == the aggregation order).
    for row in ("rowA", "rowB"):
        res = fam["tests"][row]
        t_all = np.concatenate(
            [
                np.asarray(r["t_star"])
                for r in sorted(
                    (r for r in recs if r["row"] == row), key=lambda r: r["chunk_start"]
                )
            ]
        )
        assert res["k_exceed"] == int((t_all >= res["t0"] - 1e-12).sum())
        assert res["p"] == pytest.approx((1 + res["k_exceed"]) / 81.0)


def test_bootstrap_family_not_triggered_keeps_initial_and_records_verdict(tmp_path, monkeypatch):
    monkeypatch.setattr(
        INF,
        "extension_trigger",
        lambda mc, th: {"triggered": False, "overlapping_tests": {}, "thresholds": list(th)},
    )
    fam = INF.run_family_bootstrap(
        _two_row_boot_family(tmp_path),
        2,
        BOOT_EXT_REG,
        INF.InfLedger(tmp_path / "boot2.jsonl", INF.BOOT_CHUNK_SCHEMA),
    )
    ext = fam["extension"]
    assert ext["registered"] is True and ext["fired"] is False
    assert ext["trigger"]["triggered"] is False  # verdict recorded even unfired
    assert ext["n_boot_realized"] == {"rowA": 40, "rowB": 40}


def test_studentize_degenerate_se_is_counted_not_coerced():
    t, n_deg = INF._studentize(np.array([0.2, 0.0, -0.1]), np.array([0.0, 0.0, 0.1]), delta_hat=0.0)
    assert n_deg == 2
    assert t[0] == np.inf and t[1] == 0.0 and t[2] == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Pooled-fold consumption is a loud structural dead end (unit 9 guard).
# ---------------------------------------------------------------------------
def test_pooled_fold_consumption_raises():
    fp = U.FoldPredictions(
        fold_ids=(0,),
        scores=(np.zeros(3),),
        labels=(np.array([True, False, True]),),
        prompt_ids=(np.array(["p", "p", "q"]),),
    )
    with pytest.raises(C.PooledFoldMetricError):
        fp.pooled()


# ---------------------------------------------------------------------------
# Prospective ledger: missing => REFUSE; present => denominator revision.
# ---------------------------------------------------------------------------
def test_missing_prospective_ledger_raises(tmp_path):
    with pytest.raises(INF.InferenceInputError, match="not found"):
        INF.load_prospective_ledger(tmp_path / "absent.json")
    bad = tmp_path / "manifest.json"
    bad.write_text(json.dumps({"manifest_kind": "eligible_frame", "rows": []}))
    with pytest.raises(INF.InferenceInputError, match="prospective_not_estimable_ledger"):
        INF.load_prospective_ledger(bad)


def _mini_records(tmp_path, *, seed=5):
    rd = U.synthesize_row_data(
        row="synthetic", n_prompts=60, n_responses=6, d=8, n_superfamilies=12, effect=2.0, seed=seed
    )
    records = INF._synthetic_ladder(rd, tmp_path / "comp", seed=seed)
    return rd, records


def test_ledger_drives_per_row_denominator_revision(tmp_path):
    rd, records = _mini_records(tmp_path)
    cells = sorted({f"{r.source_frame}|{r.stratum}" for r in rd.rows})
    victim = cells[-1]
    ledger = INF.synthetic_row_ledger(
        rd.row, cells, [{"cell": victim, "cause": "bank-too-small", "n_test_eligible": 3}]
    )
    comps = ["c5_full_probe", "c2_direction_dot"]
    pc = INF.prompt_cells_from_rowdata(rd)
    panel = INF.build_panel(rd.row, records, comps, ledger, pc)
    acc = panel.accounting
    assert acc["cells_total"] == 4 and acc["cells_used"] == 3
    assert acc["excluded_cells"] == [
        {"cell": victim, "cause": "bank-too-small", "n_test_eligible": 3}
    ]
    assert victim not in set(panel.cells)
    assert acc["n_rows_excluded"] > 0
    assert acc["n_prompts_kept"] == acc["n_prompts_total"] - acc["n_prompts_excluded"]
    # No exclusions => full denominator, identical row set.
    panel_full = INF.build_panel(
        rd.row, records, comps, INF.synthetic_row_ledger(rd.row, cells, []), pc
    )
    assert panel_full.accounting["cells_used"] == 4
    assert panel_full.labels.shape[0] == acc["n_rows_total"]
    # A test prompt with no cell mapping refuses loudly.
    with pytest.raises(INF.InferenceInputError, match="no cell mapping"):
        missing_pc = {k: v for k, v in list(pc.items())[1:]}
        INF.build_panel(rd.row, records, comps, ledger, missing_pc)


def test_row_gate_failure_returns_not_estimable(tmp_path):
    rd, records = _mini_records(tmp_path)
    cells = sorted({f"{r.source_frame}|{r.stratum}" for r in rd.rows})
    comps = ["c5_full_probe", "c2_direction_dot"]
    panel = INF.build_panel(
        rd.row,
        records,
        comps,
        INF.synthetic_row_ledger(rd.row, cells, []),
        INF.prompt_cells_from_rowdata(rd),
    )
    c5_rec = records[(rd.row, "c5_full_probe")]
    rows_input = {
        rd.row: INF.RowInputs(
            row=rd.row,
            panel=panel,
            rowdata=rd,
            selected_c=float(c5_rec["selected_c"]),
            scores_sha={c: records[(rd.row, c)]["scores_sha256"] for c in comps},
        )
    }
    partition = {"eligible": [rd.row], "not_estimable": []}
    # DEFAULT registry gates (>=100 discordant etc.) fail on this tiny panel.
    report = INF.run_inference(
        rows_input,
        partition,
        INF.REGISTERED_INFERENCE,
        tmp_path / "out",
        require_registered_universe=False,
    )
    assert report["rows"][rd.row]["estimable"] is False
    for fam in ("C2", "C5", "C5_minus_C2"):
        assert report["families"][fam]["tests"] == {}  # never a proxy target
        assert "row-level production gate failed" in report["not_estimable"][fam][rd.row]
    # The registered universe guard fires when required.
    with pytest.raises(INF.InferenceInputError, match="registered ROW_IDS"):
        INF.run_inference(rows_input, partition, INF.REGISTERED_INFERENCE, tmp_path / "out2")


# ---------------------------------------------------------------------------
# Checkpoint/resume: machine-stable keys, resume-skip, drift guard.
# ---------------------------------------------------------------------------
def test_resume_skips_completed_chunks_and_keys_on_generating_params(tmp_path):
    panel = _panel("rowR", seed=11)
    path = tmp_path / "perm.jsonl"
    led = INF.InfLedger(path, INF.PERM_CHUNK_SCHEMA)
    res1 = INF.run_permutation_test(
        panel,
        "c2_direction_dot",
        scores_fingerprint="s1",
        ledger=led,
        chunk_plan=TINY_REG.perm_chunk_initial,
    )
    n_recs = len(path.read_text().splitlines())
    assert n_recs == len(TINY_REG.perm_chunk_initial)
    # Fresh ledger instance, same path: every chunk resumes, nothing recomputed.
    res2 = INF.run_permutation_test(
        panel,
        "c2_direction_dot",
        scores_fingerprint="s1",
        ledger=INF.InfLedger(path, INF.PERM_CHUNK_SCHEMA),
        chunk_plan=TINY_REG.perm_chunk_initial,
    )
    assert res2 == res1
    assert len(path.read_text().splitlines()) == n_recs
    # A changed generating parameter (scores fingerprint) => new keys/records.
    INF.run_permutation_test(
        panel,
        "c2_direction_dot",
        scores_fingerprint="s2",
        ledger=INF.InfLedger(path, INF.PERM_CHUNK_SCHEMA),
        chunk_plan=TINY_REG.perm_chunk_initial,
    )
    assert len(path.read_text().splitlines()) == 2 * n_recs
    # Float generating parameters are refused (machine-stable keys, #1336).
    with pytest.raises(INF.InferenceInputError, match="machine-stable"):
        INF.InfLedger.chunk_key(
            kind="perm",
            row="r",
            comparator="c",
            scores_fingerprint="s",
            panel_fingerprint="p",
            chunk_start=0.5,
            chunk_size=10,
        )


def test_obs_stat_drift_guard_fires_on_tampered_record(tmp_path):
    panel = _panel("rowD", seed=13)
    path = tmp_path / "perm.jsonl"
    INF.run_permutation_test(
        panel,
        "c2_direction_dot",
        scores_fingerprint="s1",
        ledger=INF.InfLedger(path, INF.PERM_CHUNK_SCHEMA),
        chunk_plan=(20,),
    )
    rec = json.loads(path.read_text().splitlines()[0])
    rec["obs_stat"] += 0.25  # simulate silent input drift behind the same key
    path.write_text(json.dumps(rec) + "\n")
    with pytest.raises(C.CacheStaleError, match="input drift"):
        INF.run_permutation_test(
            panel,
            "c2_direction_dot",
            scores_fingerprint="s1",
            ledger=INF.InfLedger(path, INF.PERM_CHUNK_SCHEMA),
            chunk_plan=(20,),
        )


# ---------------------------------------------------------------------------
# Registry pins.
# ---------------------------------------------------------------------------
def test_registry_defaults_are_the_registered_plan_constants():
    reg = INF.REGISTERED_INFERENCE
    assert reg.alpha == C.HOLM["alpha"] == 0.05
    assert reg.n_perm_initial == C.HOLM["n_permutations_initial"] == 9_999
    assert reg.n_perm_extended == C.HOLM["n_permutations_extended"] == 99_999
    assert sum(reg.perm_chunk_initial) == 9_999
    assert sum(reg.perm_chunk_initial) + sum(reg.perm_chunk_extension) == 99_999
    assert (reg.min_discordant_prompts, reg.min_answers_per_class, reg.min_prompts_per_class) == (
        100,
        100,
        30,
    )
    assert reg.n_boot_extended == 20_000  # pre-registered bootstrap extension (10x)
    assert reg.n_boot_extended % reg.boot_chunk == 0 and reg.n_boot_extended > reg.n_boot
    with pytest.raises(INF.InferenceInputError):
        INF.InferenceRegistry(perm_chunk_initial=(5,))  # plan sum drifted
    with pytest.raises(INF.InferenceInputError):
        INF.InferenceRegistry(n_boot_extended=30)  # must exceed n_boot
