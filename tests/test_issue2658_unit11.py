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
- the FULL plan-§8 post-label gate set (group-J fix round): the
  complete-labels gate FAILS a silently narrowed final-label panel (and a
  passing row still records its zero counts); the label_reliability gate
  FAILS every judged row when the frozen TEST-BANK audit artifact is missing
  or failed, REFUSES a dev-side / instrument-drifted artifact, and exempts
  objective-label rows with the exemption stated; the REALIZED
  >=15-discordant-prompts-per-cell floor excludes below-floor cells with
  cause realized-discordance-below-floor, revises the denominator, and
  re-gates the reduced panel (all-below => not-estimable, never a top-up);
- build_panel REFUSES a ledger-excluded cell (n_test_eligible > 0) that
  matches no realized cell name (frame-manifest vs gen-manifest drift);
- the report phase REJECTS --n-boot/--boot-chunk overrides (smoke/measure
  keep them); the permutation MC interval uses the PASSED registry mc_conf;
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
    # Tiny realized per-cell floor so the count-gate tests keep their original
    # semantics; the realized-floor regression tests set their own floors.
    min_discordant_prompts_per_cell=1,
)


def _rel_pass(*rows: str) -> dict:
    """Synthetic PASSing test-bank reliability verdict for the named rows."""
    return {
        "status": PW.GATE_PASS,
        "artifact": "synthetic test verdict",
        "per_trait": {r: {"status": PW.GATE_PASS, "detail": "synthetic"} for r in rows},
        "bank": "test",
    }


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
            label_exclusions={"dev": {}, "test": {}},
        )
    }
    partition = {"eligible": [rd.row], "not_estimable": []}
    # COUNT gates at registered floors (>=100 discordant etc.) fail on this
    # tiny panel. per-cell floor 1 keeps the original count-gate semantics
    # (the realized-floor path has its own regression test below); this test
    # was updated for the group-J gate-set fix (new required RowInputs /
    # run_inference arguments) — the OLD three-gate set it exercised was the
    # confirmed §8 subset-implementation blocker.
    count_gate_reg = INF.InferenceRegistry(min_discordant_prompts_per_cell=1)
    report = INF.run_inference(
        rows_input,
        partition,
        count_gate_reg,
        tmp_path / "out",
        reliability=_rel_pass(rd.row),
        require_registered_universe=False,
    )
    assert report["rows"][rd.row]["estimable"] is False
    for fam in ("C2", "C5", "C5_minus_C2"):
        assert report["families"][fam]["tests"] == {}  # never a proxy target
        assert "row-level production gate failed" in report["not_estimable"][fam][rd.row]
    # The registered universe guard fires when required.
    with pytest.raises(INF.InferenceInputError, match="registered ROW_IDS"):
        INF.run_inference(
            rows_input,
            partition,
            INF.REGISTERED_INFERENCE,
            tmp_path / "out2",
            reliability=_rel_pass(rd.row),
        )


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
    assert reg.min_discordant_prompts_per_cell == 15  # plan §8 REALIZED per-cell floor
    assert reg.n_boot_extended == 20_000  # pre-registered bootstrap extension (10x)
    assert reg.n_boot_extended % reg.boot_chunk == 0 and reg.n_boot_extended > reg.n_boot
    with pytest.raises(INF.InferenceInputError):
        INF.InferenceRegistry(perm_chunk_initial=(5,))  # plan sum drifted
    with pytest.raises(INF.InferenceInputError):
        INF.InferenceRegistry(n_boot_extended=30)  # must exceed n_boot
    with pytest.raises(INF.InferenceInputError):
        INF.InferenceRegistry(min_discordant_prompts_per_cell=0)  # floor must be >= 1


# ---------------------------------------------------------------------------
# Group-J fix round: the FULL plan-§8 post-label gate set (one regression test
# per confirmed blocker + the raised build_panel assert + the override
# rejection + the mc_conf nit).
# ---------------------------------------------------------------------------
def _mini_rows_input(rd, records, panel, comps, *, label_exclusions):
    c5_rec = records[(rd.row, "c5_full_probe")]
    return {
        rd.row: INF.RowInputs(
            row=rd.row,
            panel=panel,
            rowdata=rd,
            selected_c=float(c5_rec["selected_c"]),
            scores_sha={c: records[(rd.row, c)]["scores_sha256"] for c in comps},
            label_exclusions=label_exclusions,
        )
    }


def test_complete_labels_gate_blocks_narrowed_final_panels(tmp_path):
    """Fix 1 (BLOCKER complete-labels-gate-absent): non-scored final-label
    statuses flip the row to not-estimable, and a PASSING row still records
    its zero counts."""
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
    partition = {"eligible": [rd.row], "not_estimable": []}
    # Reachable non-scored statuses (objective_labels + judge side) narrow the
    # panel: the gate FAILS and names itself, per-status counts recorded.
    rows_input = _mini_rows_input(
        rd,
        records,
        panel,
        comps,
        label_exclusions={"dev": {}, "test": {"harness_failure": 2, "malformed": 1}},
    )
    report = INF.run_inference(
        rows_input,
        partition,
        TINY_REG,
        tmp_path / "out-fail",
        reliability=_rel_pass(rd.row),
        require_registered_universe=False,
    )
    row_rep = report["rows"][rd.row]
    assert row_rep["estimable"] is False
    check = row_rep["gates"]["checks"]["complete_labels"]
    assert check["pass"] is False and check["value"] == 3
    assert check["per_split_per_status"]["test"] == {"harness_failure": 2, "malformed": 1}
    assert "complete_labels" in report["not_estimable"]["C5"][rd.row]
    assert report["families"]["C5"]["tests"] == {}  # never a proxy
    # Zero exclusions: the row passes AND still shows its zeros in the report.
    rows_input = _mini_rows_input(
        rd, records, panel, comps, label_exclusions={"dev": {}, "test": {}}
    )
    report = INF.run_inference(
        rows_input,
        partition,
        TINY_REG,
        tmp_path / "out-pass",
        reliability=_rel_pass(rd.row),
        require_registered_universe=False,
    )
    check = report["rows"][rd.row]["gates"]["checks"]["complete_labels"]
    assert check["pass"] is True and check["value"] == 0
    assert check["per_split_per_status"] == {"dev": {}, "test": {}}
    assert report["rows"][rd.row]["estimable"] is True
    # A mis-threaded diag (missing split) refuses loudly.
    _, floor_check = INF.apply_realized_cell_floor(panel, TINY_REG)
    with pytest.raises(INF.InferenceInputError, match="dev\\+test"):
        INF.row_gates(
            panel,
            TINY_REG,
            label_exclusions={"test": {}},
            label_source="synthetic",
            reliability=_rel_pass(rd.row),
            realized_cell_floor=floor_check,
        )


def _write_test_audit(tmp_path, body):
    p = tmp_path / INF.TEST_AUDIT_DIR / PW.HUMAN_AUDIT_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body))
    return p


def test_label_reliability_missing_artifact_blocks_judged_rows(tmp_path):
    """Fix 2 (BLOCKER reliability-gate-absent): a MISSING test-bank audit
    artifact refuses estimability for every judged row (never a pass)."""
    verdict = INF.load_test_label_reliability(tmp_path)
    assert verdict["status"] == PW.GATE_NOT_ESTIMABLE and verdict["per_trait"] == {}
    assert verdict["missing_artifact"].endswith("human_audit_test/human_audit/adjudications.json")
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
    rows_input = _mini_rows_input(
        rd, records, panel, comps, label_exclusions={"dev": {}, "test": {}}
    )
    report = INF.run_inference(
        rows_input,
        {"eligible": [rd.row], "not_estimable": []},
        TINY_REG,
        tmp_path / "out",
        reliability=verdict,
        require_registered_universe=False,
    )
    row_rep = report["rows"][rd.row]
    assert row_rep["estimable"] is False
    check = row_rep["gates"]["checks"]["label_reliability"]
    assert check["pass"] is False and check["status"] == "MISSING"
    assert "label_reliability" in report["not_estimable"]["C5"][rd.row]
    assert report["label_reliability"]["status"] == PW.GATE_NOT_ESTIMABLE


def test_label_reliability_refuses_devside_or_drifted_artifact(tmp_path):
    """Fix 2: a PRESENT artifact with dev-side / missing / drifted provenance
    RAISES — a mis-wired audit is never read as an unrun one."""
    judged = [r for r in C.ROW_IDS if C.CONSTRUCTS[r].judge_scored]
    # No bank marker (the dev artifact's shape) and an explicit dev marker.
    _write_test_audit(tmp_path, {"rows": []})
    with pytest.raises(INF.InferenceInputError, match="bank"):
        INF.load_test_label_reliability(tmp_path)
    _write_test_audit(tmp_path, {"bank": "dev", "rows": []})
    with pytest.raises(INF.InferenceInputError, match="dev-side"):
        INF.load_test_label_reliability(tmp_path)
    # bank=test but no fingerprints envelope.
    _write_test_audit(tmp_path, {"bank": "test", "rows": []})
    with pytest.raises(INF.InferenceInputError, match="judge_instrument_fingerprints"):
        INF.load_test_label_reliability(tmp_path)
    # Drifted judge-instrument fingerprint.
    _write_test_audit(
        tmp_path,
        {"bank": "test", "judge_instrument_fingerprints": {judged[0]: "deadbeef"}, "rows": []},
    )
    with pytest.raises(INF.InferenceInputError, match="judge-instrument"):
        INF.load_test_label_reliability(tmp_path)
    # Valid envelope: the verdict flows through the IMPORTED power machinery
    # (zero adjudication rows => NOT-ESTIMABLE per_trait, honest and empty).
    fps = {r: C.judge_instrument_fingerprint(r) for r in judged}
    _write_test_audit(tmp_path, {"bank": "test", "judge_instrument_fingerprints": fps, "rows": []})
    verdict = INF.load_test_label_reliability(tmp_path)
    assert verdict["status"] == PW.GATE_NOT_ESTIMABLE and verdict["bank"] == "test"
    assert verdict["per_trait"] == {}


def test_label_reliability_objective_rows_exempt_with_stated_exemption(tmp_path):
    """Fix 2: objective-label (correctness) rows are exempt, and the exemption
    is STATED in the gate record — never a silent skip."""
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
    _, floor_check = INF.apply_realized_cell_floor(panel, TINY_REG)
    missing = {"status": PW.GATE_NOT_ESTIMABLE, "per_trait": {}, "missing_artifact": "absent"}
    gates = INF.row_gates(
        panel,
        TINY_REG,
        label_exclusions={"dev": {}, "test": {}},
        label_source="objective-labels",
        reliability=missing,
        realized_cell_floor=floor_check,
    )
    rel = gates["checks"]["label_reliability"]
    assert rel["pass"] is True and "exempt" in rel and "objective labels" in rel["exempt"]
    # The same missing verdict FAILS a judged/synthetic row.
    gates = INF.row_gates(
        panel,
        TINY_REG,
        label_exclusions={"dev": {}, "test": {}},
        label_source="synthetic",
        reliability=missing,
        realized_cell_floor=floor_check,
    )
    assert gates["checks"]["label_reliability"]["pass"] is False
    with pytest.raises(INF.InferenceInputError, match="unknown label_source"):
        INF.row_gates(
            panel,
            TINY_REG,
            label_exclusions={"dev": {}, "test": {}},
            label_source="mystery",
            reliability=missing,
            realized_cell_floor=floor_check,
        )


def _two_cell_panel(row="floorrow", *, b_discordant=True):
    """Hand-built final-label panel: cell A holds 6 discordant prompts; cell B
    holds 2 prompts, discordant or concordant per ``b_discordant``."""
    pids: list[str] = []
    labels: list[bool] = []
    cells: list[str] = []
    for i in range(6):
        pids += [f"{row}-A{i}"] * 2
        labels += [True, False]
        cells += ["frameA|band0"] * 2
    for i in range(2):
        pids += [f"{row}-B{i}"] * 2
        labels += [True, False] if b_discordant else [True, True]
        cells += ["frameB|band0"] * 2
    n = len(pids)
    rng = np.random.default_rng(3)
    return INF.InferencePanel(
        row=row,
        prompt_ids=np.array(pids),
        response_index=np.tile(np.arange(2, dtype=np.int64), n // 2),
        labels=np.array(labels, dtype=bool),
        scores={"c5_full_probe": rng.standard_normal(n)},
        cells=np.array(cells),
        accounting={
            "floor": 15,
            "cells_total": 2,
            "cells_used": 2,
            "excluded_cells": [],
            "n_rows_total": n,
            "n_rows_kept": n,
            "n_rows_excluded": 0,
            "n_prompts_total": 8,
            "n_prompts_kept": 8,
            "n_prompts_excluded": 0,
        },
    )


def test_realized_cell_floor_excludes_and_revises_denominator(tmp_path):
    """Fix 3 (BLOCKER realized-cell-floor-unchecked): per-RETAINED-cell
    discordant-prompt counts are computed AFTER final labels; a below-floor
    cell is excluded with cause realized-discordance-below-floor, the
    denominator is revised, the row gates RE-RUN on the reduced panel, and no
    test row is ever added."""
    reg = INF.InferenceRegistry(
        min_discordant_prompts=7,
        min_answers_per_class=1,
        min_prompts_per_class=1,
        min_discordant_prompts_per_cell=3,
    )
    # Zero-realized-discordance cell (the reconciler's fatal example): cell B
    # realizes 0 discordant prompts and is EXCLUDED, never flowing into the
    # macro estimate.
    panel = _two_cell_panel(b_discordant=False)
    reduced, check = INF.apply_realized_cell_floor(panel, reg)
    assert check["pass"] is True and check["per_cell_discordant"] == {
        "frameA|band0": 6,
        "frameB|band0": 0,
    }
    assert check["excluded"] == [
        {"cell": "frameB|band0", "cause": "realized-discordance-below-floor", "n_discordant": 0}
    ]
    acc = reduced.accounting
    assert acc["realized_excluded_cells"] == check["excluded"]
    assert acc["cells_used"] == 1 and acc["n_rows_kept"] == 12 and acc["n_rows_excluded"] == 4
    assert acc["n_prompts_kept"] == 6 and acc["n_prompts_excluded"] == 2
    # No new test rows: the reduced panel is a strict subset of the original.
    assert set(reduced.prompt_ids) < set(panel.prompt_ids)
    assert reduced.labels.shape[0] < panel.labels.shape[0]
    # Re-gating bites on the REDUCED panel: 6+2=8 discordant would pass the
    # >=7 row floor, but cell B (2 < 3 per-cell) is excluded first and the
    # reduced 6 fails it.
    panel = _two_cell_panel(b_discordant=True)
    reduced, check = INF.apply_realized_cell_floor(panel, reg)
    assert check["per_cell_discordant"]["frameB|band0"] == 2 and check["cells_below_floor"] == 1
    gates = INF.row_gates(
        reduced,
        reg,
        label_exclusions={"dev": {}, "test": {}},
        label_source="objective-labels",
        reliability={"per_trait": {}},
        realized_cell_floor=check,
    )
    assert gates["checks"]["discordant_prompts"]["value"] == 6
    assert gates["checks"]["discordant_prompts"]["pass"] is False
    assert gates["estimable"] is False
    # ALL cells below the floor => not-estimable outright (never a top-up).
    reduced, check = INF.apply_realized_cell_floor(panel, INF.REGISTERED_INFERENCE)
    assert reduced is None and check["pass"] is False and check["cells_below_floor"] == 2
    # Integration: the production-floor short circuit names the gate in the
    # report and feeds no family a proxy target.
    rd, records = _mini_records(tmp_path)
    cells = sorted({f"{r.source_frame}|{r.stratum}" for r in rd.rows})
    comps = ["c5_full_probe", "c2_direction_dot"]
    mini_panel = INF.build_panel(
        rd.row,
        records,
        comps,
        INF.synthetic_row_ledger(rd.row, cells, []),
        INF.prompt_cells_from_rowdata(rd),
    )
    rows_input = _mini_rows_input(
        rd, records, mini_panel, comps, label_exclusions={"dev": {}, "test": {}}
    )
    report = INF.run_inference(
        rows_input,
        {"eligible": [rd.row], "not_estimable": []},
        INF.REGISTERED_INFERENCE,
        tmp_path / "out",
        reliability=_rel_pass(rd.row),
        require_registered_universe=False,
    )
    assert report["rows"][rd.row]["estimable"] is False
    assert "realized_cell_floor" in report["not_estimable"]["C5"][rd.row]
    assert report["families"]["C5"]["tests"] == {}


def test_report_phase_rejects_bootstrap_overrides(tmp_path):
    """Fix 4 (override hardening): --n-boot/--boot-chunk are refused in the
    confirmatory report phase and kept for the smoke/measure phases."""
    ap = INF.build_argparser()
    for extra in (["--n-boot", "40"], ["--boot-chunk", "20"]):
        args = ap.parse_args(["--phase", "report", "--comparators-dir", str(tmp_path), *extra])
        with pytest.raises(INF.InferenceInputError, match="registered constants"):
            INF.run(args)
    # The measure/smoke phases keep the recorded override seam.
    args = ap.parse_args(["--phase", "measure-boot-unit", "--n-boot", "40", "--boot-chunk", "20"])
    reg = INF._registry_from_args(args)
    assert reg.n_boot == 40 and reg.boot_chunk == 20


def test_build_panel_refuses_ghost_excluded_cell(tmp_path):
    """Fix 5 (raised CONCERN ledger-exclusion-noop-unasserted): a ledger-
    excluded cell with eligible test prompts that matches NO realized cell
    name (frame-manifest vs gen-manifest drift) refuses loudly instead of
    silently readmitting the cell."""
    rd, records = _mini_records(tmp_path)
    cells = sorted({f"{r.source_frame}|{r.stratum}" for r in rd.rows})
    comps = ["c5_full_probe", "c2_direction_dot"]
    pc = INF.prompt_cells_from_rowdata(rd)
    ghost = INF.synthetic_row_ledger(
        rd.row,
        cells,
        [{"cell": "frameB|band-1-drifted", "cause": "bank-too-small", "n_test_eligible": 3}],
    )
    with pytest.raises(INF.InferenceInputError, match="match NO realized"):
        INF.build_panel(rd.row, records, comps, ghost, pc)
    # A ghost with ZERO eligible test prompts is legitimate (nothing to
    # exclude) — no refusal, full denominator.
    for rec in (
        {"cell": "frameB|band-1-drifted", "cause": "bank-too-small", "n_test_eligible": 0},
        {"cell": "frameB|band-1-drifted", "cause": "bank-too-small"},
    ):
        benign = INF.synthetic_row_ledger(rd.row, cells, [rec])
        panel = INF.build_panel(rd.row, records, comps, benign, pc)
        assert panel.accounting["n_rows_excluded"] == 0


def test_permutation_mc_interval_uses_passed_mc_conf(tmp_path):
    """Fix 6 (NIT): the permutation MC interval reads the PASSED mc_conf,
    not the module-level registry."""
    panel = _panel("mcconf", n_prompts=8, m=4)
    ledger = INF.InfLedger(tmp_path / "perm.jsonl", INF.PERM_CHUNK_SCHEMA)
    common = dict(scores_fingerprint="sha-mc", ledger=ledger, chunk_plan=(20, 20))
    res_lo = INF.run_permutation_test(panel, "c2_direction_dot", mc_conf=0.5, **common)
    res_hi = INF.run_permutation_test(panel, "c2_direction_dot", mc_conf=0.99, **common)
    assert res_lo["mc_interval"] == pytest.approx(
        INF.clopper_pearson_interval(res_lo["k_exceed"], res_lo["n_perm"], 0.5)
    )
    assert res_hi["mc_interval"] == pytest.approx(
        INF.clopper_pearson_interval(res_hi["k_exceed"], res_hi["n_perm"], 0.99)
    )
    lo_lo, lo_hi = res_lo["mc_interval"]
    hi_lo, hi_hi = res_hi["mc_interval"]
    assert hi_lo <= lo_lo and lo_hi <= hi_hi and (lo_lo, lo_hi) != (hi_lo, hi_hi)


# ---------------------------------------------------------------------------
# Round 12 (plan v5 A1): owner-ruling waiver on the test-bank reliability path.
# ---------------------------------------------------------------------------
def _waiver_fixture_v5() -> dict:
    return {
        "schema": PW.WAIVER_SCHEMA,
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


def _write_waiver_v5(tmp_path, body) -> None:
    p = tmp_path / PW.HUMAN_AUDIT_WAIVER_REL
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(body))


def test_label_reliability_waived_by_owner_ruling(tmp_path):
    """v5 A1: no test-bank adjudications + a valid waiver record => WAIVED for
    every judged row (never PASS), the disclosure travels, and the row gate
    treats WAIVED as pass=True with the disclosure in detail."""
    _write_waiver_v5(tmp_path, _waiver_fixture_v5())
    verdict = INF.load_test_label_reliability(tmp_path)
    judged = [r for r in C.ROW_IDS if C.CONSTRUCTS[r].judge_scored]
    assert verdict["status"] == PW.GATE_WAIVED and verdict["bank"] == "test"
    assert sorted(verdict["per_trait"]) == sorted(judged)
    for v in verdict["per_trait"].values():
        assert v["status"] == PW.GATE_WAIVED and "SYNTHETIC-DISCLOSURE" in v["detail"]
    assert verdict["waiver"]["ruling_verbatim"] == "just trust the judge"
    # the row gate: WAIVED passes with the disclosure in detail, status verbatim
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
    rows_input = _mini_rows_input(
        rd, records, panel, comps, label_exclusions={"dev": {}, "test": {}}
    )
    report = INF.run_inference(
        rows_input,
        {"eligible": [rd.row], "not_estimable": []},
        TINY_REG,
        tmp_path / "out",
        reliability=verdict,
        require_registered_universe=False,
    )
    check = report["rows"][rd.row]["gates"]["checks"]["label_reliability"]
    assert check["pass"] is True
    assert check["status"] == PW.GATE_WAIVED
    assert "SYNTHETIC-DISCLOSURE" in check["detail"]
    assert "label_reliability" not in report["not_estimable"]["C5"].get(rd.row, [])


def test_label_reliability_malformed_waiver_raises(tmp_path):
    bad = _waiver_fixture_v5()
    bad["ruling_event"]["kind"] = "epm:progress"
    _write_waiver_v5(tmp_path, bad)
    with pytest.raises(PW.PowerInputError, match="malformed waiver"):
        INF.load_test_label_reliability(tmp_path)


def test_label_reliability_waiver_ignored_when_test_audit_exists(tmp_path):
    """A REAL test-bank audit wins over the waiver: the envelope validation and
    the real reliability machinery run (empty rows => honest NOT-ESTIMABLE)."""
    _write_waiver_v5(tmp_path, _waiver_fixture_v5())
    judged = [r for r in C.ROW_IDS if C.CONSTRUCTS[r].judge_scored]
    fps = {r: C.judge_instrument_fingerprint(r) for r in judged}
    _write_test_audit(tmp_path, {"bank": "test", "judge_instrument_fingerprints": fps, "rows": []})
    verdict = INF.load_test_label_reliability(tmp_path)
    assert verdict["status"] == PW.GATE_NOT_ESTIMABLE  # real machinery, not WAIVED
    assert verdict["per_trait"] == {}
