"""arm8 + arm12 + arm20 in the Result-2 FAIR-PROTOCOL v2 roster — #1739 (2026-08-06).

The fair re-score (scripts/issue1739_result2fair_score.py) originally carried
six arms across a linear + an mlp pass; the 2026-08-06 follow-ups add THREE
arms and drop MLP entirely:

- arm12_oracle_reg ("ridge regression on the real answer" — the fitted
  oracle), the one five-method-comparison method R2FAIR never ran;
- arm20_shuffled_map_ridge (ridge on the SHUFFLED-weight mapped answer) — the
  fitted-readout counterpart of arm13's projection control, a CONTROL /
  falsification test of the linear-collapse argument, never a method;
- arm8_map_ridge_true — arm12's fitted w APPLIED TO THE MAPPED answer (one za
  RidgeJob, two eval matrices), the map-error-sensitive regression comparator
  (arm8:arm12 is the fitted analogue of arm6:arm11); ZERO extra
  factorizations by construction;
- the map_kind="mlp" pass AND the MLP readout (arm19_map_mlp_pred) are
  DROPPED (user scope decision 2026-08-06) — the engine keeps arm19 in
  ARM_REGISTRY; only the fair roster loses it. Output moves to the
  result2_fair_v2/ sibling so the committed result2_fair/ tree (source of
  already-shipped figures) is never overwritten.

Pin groups:

1. arm12 in the LINEAR roster, COMMITTED_FROZEN_ARMS, LABEL_CONSUMING.
   arm20 in the LINEAR roster and LABEL_CONSUMING but deliberately NOT in
   COMMITTED_FROZEN_ARMS (new arm, no committed rows: own-train-OOF argmax).
   arm8 in all three (its committed-modal layers verified live:
   evil=21/syco=17/hall=16).
2. NO MLP anywhere: ROSTER_BY_KIND is linear-only, ROSTER_MLPMAP is gone,
   --map-kinds accepts only "linear", no mlp/arm19 slots in the fig.
3. The fair transfer path actually SCORES arm8/arm12/arm20 (the za and mps
   RidgeJobs built and scattered) when the linear roster is requested.
4. arm13 and arm20 consume ONE identical shuffle per cell: exactly one
   shuffled_map_weights draw (seed=cell.seed) serves both controls, and an
   explicitly-provided w_shuffled suppresses the draw entirely.
5. Points-file slots are their own names; arm11's/arm12's real-answer slots
   are exempt from the reliability ceiling; arm8's + arm20's slots ARE
   ceiling-bounded (deterministic functions of the context) inside the
   CONTIGUOUS bounded prefix; arm8's label makes the fit-on-REAL /
   apply-to-MAPPED split unmistakable; arm20's label says "control".
6. Batch-1 sequencing: the mps (arm 20) ridge jobs solve in their OWN batch
   BEFORE the main z/mp/za batch, and mp_shuf is released with a
   refcount-verified guard + INFO log (the fix-engaged signal); a retained
   reference trips the guard (RuntimeError).
7. Matched-layer companions: arm20 rows carry rho_matched_arm7_layer and
   arm8 rows carry rho_matched_arm12_layer (+ matched_layer /
   matched_layer_idx / n_eval_matched / matched_note) — each arm's rho at
   its reference arm's frozen layer, an index into the scored profile;
   fails loud when the reference layer is absent while the arm's rows
   exist; a no-op when no rows exist.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_1739 import arms, fits

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue1739_result2fair_fig as fair_fig  # noqa: E402
import issue1739_result2fair_score as fair_score  # noqa: E402

SLUG = "arm12_oracle_reg"
SLUG8 = "arm8_map_ridge_true"
SLUG20 = "arm20_shuffled_map_ridge"


def _toy_mapfit(*, n_layers: int = 2, dim: int = 6, seed: int = 0) -> fits.MapFit:
    rng = np.random.default_rng(seed)
    return fits.MapFit(
        w=rng.standard_normal((n_layers, dim, dim)) * 0.1,
        x_mu=np.zeros((n_layers, 1, dim)),
        x_sd=np.ones((n_layers, 1, dim)),
        y_mu=np.zeros((n_layers, 1, dim)),
        diagnostics={},
        kind="linear",
    )


def _toy_cell(*, seed: int = 0):
    """A tiny synthetic transfer cell: (data, cell, z_ev, dv_ev, za_ev)."""
    rng = np.random.default_rng(seed)
    n_layers, n_train, n_eval, dim = 2, 40, 20, 6
    data = arms.CellData(
        z_ctx=rng.standard_normal((n_layers, n_train, dim)),
        z_ans=rng.standard_normal((n_layers, n_train, dim)),
        dv=rng.uniform(0, 100, n_train),
        rb=rng.standard_normal((n_layers, dim)),
        mapfit=_toy_mapfit(n_layers=n_layers, dim=dim),
        layers=tuple(range(n_layers)),
    )
    cell = fits.realize_budget_cell(np.arange(n_train) % 5, budget_l=n_train, draw=0, seed=0)
    return (
        data,
        cell,
        rng.standard_normal((n_layers, n_eval, dim)),
        rng.uniform(0, 100, n_eval),
        rng.standard_normal((n_layers, n_eval, dim)),
    )


def test_arm12_roster_membership():
    """arm12 in the linear roster + both conventions."""
    assert SLUG in fair_score.ROSTER_LINEAR
    assert SLUG in fair_score.COMMITTED_FROZEN_ARMS
    assert SLUG in fair_score.LABEL_CONSUMING
    # rb-independent oracle read: no regime direction anywhere in its scores.
    assert arms.ARM_REGISTRY[SLUG]["rb_dep"] is False
    assert arms.ARM_REGISTRY[SLUG]["family"] == "oracle"


def test_arm8_roster_membership():
    """arm8 in the linear roster, LABEL_CONSUMING (shares arm12's arm8_12
    DV-fitted target), AND COMMITTED_FROZEN_ARMS (committed rows verified
    live: modal layers evil=21/syco=17/hall=16 — footing parity with
    arm11/arm12)."""
    assert SLUG8 in fair_score.ROSTER_LINEAR
    assert SLUG8 in fair_score.LABEL_CONSUMING
    assert SLUG8 in fair_score.COMMITTED_FROZEN_ARMS
    spec = arms.ARM_REGISTRY[SLUG8]
    assert spec["family"] == "map"
    assert spec["layered"] is True
    # one fitted w on real answers, read on the mapped answer: rb-INDEPENDENT.
    assert spec["rb_dep"] is False


def test_arm20_roster_membership():
    """arm20 in the linear roster + LABEL_CONSUMING; NOT committed-frozen
    (new arm — the arm19 precedent: own-train-OOF argmax)."""
    assert SLUG20 in fair_score.ROSTER_LINEAR
    assert SLUG20 in fair_score.LABEL_CONSUMING
    assert SLUG20 not in fair_score.COMMITTED_FROZEN_ARMS
    spec = arms.ARM_REGISTRY[SLUG20]
    assert spec["family"] == "control"
    assert spec["layered"] is True
    # ridge mp_shuf -> dv involves no regime direction: rb-INDEPENDENT.
    assert spec["rb_dep"] is False


def test_no_mlp_anywhere():
    """The MLP drop (user scope decision 2026-08-06) is total on the fair
    surface: linear-only roster map, no ROSTER_MLPMAP symbol, --map-kinds
    accepts only linear, no mlp/arm19 slots in the fig — while the ENGINE
    keeps arm19 in ARM_REGISTRY (only the fair roster loses it)."""
    assert fair_score.ROSTER_BY_KIND == {"linear": fair_score.ROSTER_LINEAR}
    assert not hasattr(fair_score, "ROSTER_MLPMAP")
    assert "arm19_map_mlp_pred" not in fair_score.ROSTER_LINEAR
    assert "arm19_map_mlp_pred" not in fair_score.LABEL_CONSUMING
    args = fair_score.parse_args([])
    assert args.map_kinds == ["linear"]
    assert not any(kind == "mlp" for _arm, kind in fair_fig.METHOD_OF)
    assert not any(arm == "arm19_map_mlp_pred" for arm, _kind in fair_fig.METHOD_OF)
    assert not any("mlp" in slot for slot in fair_fig.SLOTS)
    assert "arm19_map_mlp_pred" not in fair_fig.FAIR_READOUT_ARMS
    assert "arm19_map_mlp_pred" in arms.ARM_REGISTRY  # engine untouched


def test_fair_rosters_keep_registry_order():
    """Roster ordering stays the ARM_REGISTRY order (the resolve_transfer_roster
    convention), so arm8 slots between arm7 and arm11."""
    order = list(arms.ARM_REGISTRY)
    idx = [order.index(a) for a in fair_score.ROSTER_LINEAR]
    assert idx == sorted(idx), f"roster not in registry order: {fair_score.ROSTER_LINEAR}"


def test_linear_roster_scores_new_arms_on_toy_transfer_cell():
    """The fair LINEAR roster produces arm8, arm12 AND arm20 scores (the za
    job with BOTH eval matrices + the mps job built and scattered) with zero
    skips — the exact call shape of the fair transfer legs
    (run_transfer_cell, ridge_folds=(0,), za_ev threaded)."""
    data, cell, z_ev, dv_ev, za_ev = _toy_cell()
    scores, skipped = arms.run_transfer_cell(
        data,
        cell,
        z_ev,
        dv_ev,
        za_ev=za_ev,
        arms=list(fair_score.ROSTER_LINEAR),
        ridge_folds=(0,),
    )
    assert not skipped, f"fair linear roster skipped arms: {skipped}"
    assert set(scores) == set(fair_score.ROSTER_LINEAR)
    for slug in (SLUG, SLUG8, SLUG20):
        assert scores[slug].shape == (2, 20), slug
        assert np.isfinite(scores[slug]).all(), slug
    # arm8 and arm12 share ONE fitted w but read DIFFERENT eval matrices —
    # the scores must differ (identical scores would mean the eval split is
    # broken and both read the same input).
    assert not np.array_equal(scores[SLUG8], scores[SLUG])


def _toy_datas(n: int = 30, d: int = 6, ly: int = 2, seed: int = 0):
    """Two regime slices sharing every rb-independent input (run_cell_multi's
    identity contract), differing only in rb — the arm19-test convention."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((ly, n, d))
    za = z + 0.3 * rng.standard_normal((ly, n, d))
    dv = rng.uniform(0, 100, n)
    mf = _toy_mapfit(n_layers=ly, dim=d)
    datas = [
        arms.CellData(
            z_ctx=z,
            z_ans=za,
            dv=dv,
            rb=rng.standard_normal((ly, d)),
            mapfit=mf,
            layers=tuple(range(ly)),
        )
        for _ in range(2)
    ]
    cell = fits.realize_budget_cell(np.arange(n) % 5, budget_l=n, draw=0, seed=0)
    return datas, cell


def test_arm13_arm20_share_one_shuffle(monkeypatch):
    """BOTH shuffled-weight controls read ONE identical shuffle per cell.

    Pins the fair-roster hoist: exactly one shuffled_map_weights draw
    (seed=cell.seed) serves arm13's projection AND arm20's ridge; arm13 stays
    rb-dependent (distinct scores per regime, all derived from the shared
    mp_shuf) while arm20 is rb-independent (SAME ndarray across regimes); and
    arm13's scores reproduce bitwise from an independently-derived mp_shuf at
    the same seed — the projection and ridge controls are read off one shuffle.
    """
    datas, cell = _toy_datas()
    real = fits.shuffled_map_weights
    calls: list[int] = []

    def counting(w, *, seed):
        calls.append(seed)
        return real(w, seed=seed)

    monkeypatch.setattr(arms, "shuffled_map_weights", counting)
    outs = arms.run_cell_multi(datas, cell, arms=["arm13_shuffled_map", SLUG20], device="cpu")
    (s0, sk0), (s1, sk1) = outs
    assert not sk0 and not sk1, (sk0, sk1)
    assert calls == [cell.seed], f"expected exactly ONE shuffle draw, got {calls}"
    # rb_dep contract: arm20 shared by identity, arm13 per regime
    assert s0[SLUG20] is s1[SLUG20]
    assert s0["arm13_shuffled_map"] is not s1["arm13_shuffled_map"]
    assert s0[SLUG20].shape == (2, 30)
    assert np.isfinite(s0[SLUG20]).all()
    # the shared tensor IS the seed-cell shuffle: arm13's projection reproduces
    # bitwise from mp_shuf re-derived independently at seed=cell.seed
    # (budget_l == n makes cell.row_idx the identity, so z inside run_cell_multi
    # aliases z_ctx and the reference computation sees the same bytes)
    assert np.array_equal(cell.row_idx, np.arange(30))
    w_shuf = real(datas[0].mapfit.w, seed=cell.seed)
    mp_shuf = fits.apply_map(datas[0].z_ctx, datas[0].mapfit, w=w_shuf)
    for r, (s, _sk) in enumerate(outs):
        expected = np.einsum(
            "lnd,ld->ln", mp_shuf, np.asarray(datas[r].rb, dtype=np.float64), optimize=True
        )
        assert np.array_equal(s["arm13_shuffled_map"], expected), f"regime {r}"


def test_provided_w_shuffled_suppresses_the_draw(monkeypatch):
    """An explicit CellData.w_shuffled is honored by BOTH controls — zero
    shuffled_map_weights draws, same scores as the derived-shuffle run."""
    datas, cell = _toy_datas()
    ref_outs = arms.run_cell_multi(datas, cell, arms=["arm13_shuffled_map", SLUG20], device="cpu")
    w_shuf = fits.shuffled_map_weights(datas[0].mapfit.w, seed=cell.seed)
    datas_w = [
        arms.CellData(
            z_ctx=d.z_ctx,
            z_ans=d.z_ans,
            dv=d.dv,
            rb=d.rb,
            mapfit=d.mapfit,
            w_shuffled=w_shuf,  # SAME object on both (identity-asserted upstream)
            layers=d.layers,
        )
        for d in datas
    ]

    def boom(w, *, seed):  # any draw is a contract violation
        raise AssertionError("shuffled_map_weights must not be called when w_shuffled is given")

    monkeypatch.setattr(arms, "shuffled_map_weights", boom)
    outs = arms.run_cell_multi(datas_w, cell, arms=["arm13_shuffled_map", SLUG20], device="cpu")
    for (s, sk), (rs, _rsk) in zip(outs, ref_outs, strict=True):
        assert not sk
        assert np.array_equal(s["arm13_shuffled_map"], rs["arm13_shuffled_map"])
        assert np.allclose(s[SLUG20], rs[SLUG20])


def test_no_mapfit_skips_both_shuffled_controls():
    """No mapfit ⇒ arm13 AND arm20 are SKIPPED with a recorded reason —
    never zero-filled (the arms.py recorded-skip contract)."""
    datas, cell = _toy_datas()
    data = arms.CellData(
        z_ctx=datas[0].z_ctx,
        z_ans=datas[0].z_ans,
        dv=datas[0].dv,
        rb=datas[0].rb,
        mapfit=None,
        layers=datas[0].layers,
    )
    scores, skipped = arms.run_cell(data, cell, arms=["arm13_shuffled_map", SLUG20], device="cpu")
    for slug in ("arm13_shuffled_map", SLUG20):
        assert slug in skipped and "no mapfit" in skipped[slug]
        assert slug not in scores


def test_arm20_slot_is_a_bounded_control():
    """arm20's points-file slot: its own name, ceiling-BOUNDED (a shuffled-map
    answer is a deterministic function of the context), labelled as a control."""
    slot = fair_fig.METHOD_OF[(SLUG20, "linear")]
    assert slot == "regression_shuffled_map"
    assert slot != "oracle"
    assert slot in fair_fig.SLOTS
    assert SLUG20 in fair_fig.FAIR_READOUT_ARMS
    assert slot in fair_fig.CEILING_BOUNDED
    assert slot not in fair_fig.REAL_ANSWER_SLOTS
    label = {m: lbl for _t, ms in fair_fig.GROUPS for m, lbl, _c, _h in ms}[slot]
    assert "control" in label.lower(), "the bar must never be read as a method"


def test_points_file_method_slot_distinct_from_oracle():
    """The fig emits arm12 under its OWN slot — never arm11's `oracle`."""
    slot = fair_fig.METHOD_OF[(SLUG, "linear")]
    assert slot == "regression_real_answer"
    assert slot != "oracle"
    assert slot in fair_fig.SLOTS
    # one slot per (arm, kind) — no slot collisions across the whole map
    assert len(set(fair_fig.METHOD_OF.values())) == len(fair_fig.METHOD_OF)
    # label-consuming readout field parity with arm4/7/19
    assert SLUG in fair_fig.FAIR_READOUT_ARMS
    assert set(fair_fig.FAIR_READOUT_ARMS) == set(fair_score.LABEL_CONSUMING)


def test_real_answer_slots_exempt_from_reliability_ceiling():
    """Both real-answer reads (arm11 projection, arm12 regression) share
    information with the DV's judge noise — the sqrt(r_yy) ceiling bounds
    neither; every context-based slot stays bounded."""
    assert set(fair_fig.REAL_ANSWER_SLOTS) == {"oracle", "regression_real_answer"}
    for slot in fair_fig.REAL_ANSWER_SLOTS:
        assert slot not in fair_fig.CEILING_BOUNDED
    assert set(fair_fig.CEILING_BOUNDED) == set(fair_fig.SLOTS) - set(fair_fig.REAL_ANSWER_SLOTS)
    # the rendered ceiling segment spans a CONTIGUOUS slot prefix (render()
    # draws min..max over bounded indices — a real-answer slot inside that
    # span would sit under a ceiling that does not bound it)
    bounded_idx = [fair_fig.SLOTS.index(m) for m in fair_fig.CEILING_BOUNDED]
    assert bounded_idx == list(range(len(bounded_idx)))


# ---------------------------------------------------------------------------
# batch-1 sequencing fix + matched-layer companion (team-lead round 4)
# ---------------------------------------------------------------------------


def test_mps_batch_solves_first_and_releases_mp_shuf(monkeypatch, caplog):
    """The arm-20 (mps) ridge jobs solve in their OWN batch BEFORE the main
    z/mp/za batch, and mp_shuf is released (refcount-verified, INFO-logged)
    before batch 2 — the fix-engaged signal for the sequencing fix. The
    recorder captures SOURCE NAMES only: retaining the job objects would
    (correctly) trip the release guard itself."""
    import logging

    real = arms._solve_ridge_groups
    call_sources: list[list[str]] = []

    def recording(jobs, **kw):
        call_sources.append(sorted({j.key[0] for j in jobs}))
        return real(jobs, **kw)

    monkeypatch.setattr(arms, "_solve_ridge_groups", recording)
    data, cell, z_ev, dv_ev, za_ev = _toy_cell()
    with caplog.at_level(logging.INFO, logger=arms.logger.name):
        scores, skipped = arms.run_transfer_cell(
            data,
            cell,
            z_ev,
            dv_ev,
            za_ev=za_ev,
            arms=list(fair_score.ROSTER_LINEAR),
            ridge_folds=(0,),
        )
    assert not skipped, skipped
    assert len(call_sources) == 2, call_sources
    assert call_sources[0] == ["mps"], f"batch 1 must be mps-only: {call_sources}"
    assert "mps" not in call_sources[1], call_sources
    assert {"z", "za"} <= set(call_sources[1]), call_sources
    assert np.isfinite(scores[SLUG20]).all()
    released = [r for r in caplog.records if "mp_shuf released" in r.getMessage()]
    assert released, "the batch-1 release log line (fix-engaged signal) must fire"


def test_release_guard_trips_on_a_retained_reference(monkeypatch):
    """A retained mps RidgeJob (holding mp_shuf) makes the release guard raise
    — the guard actually EXERCISES the invariant, not just an import."""
    import pytest

    real = arms._solve_ridge_groups
    stash: list = []

    def stealing(jobs, **kw):
        if jobs and jobs[0].key[0] == "mps":
            stash.append(jobs[0])  # retain a reference past the batch-1 clear
        return real(jobs, **kw)

    monkeypatch.setattr(arms, "_solve_ridge_groups", stealing)
    data, cell, z_ev, dv_ev, za_ev = _toy_cell()
    with pytest.raises(RuntimeError, match="mp_shuf still referenced"):
        arms.run_transfer_cell(
            data,
            cell,
            z_ev,
            dv_ev,
            za_ev=za_ev,
            arms=list(fair_score.ROSTER_LINEAR),
            ridge_folds=(0,),
        )


def test_matched_layer_fields_attached_and_correct():
    """attach_matched_layer_companion writes the companion FIELDS on the arm's
    rows — rho at the reference arm's frozen layer equals a direct spearman on
    that layer's score row; other arms' rows stay untouched; a below-min_n
    rung records None."""
    rng = np.random.default_rng(3)
    n = 12
    sc = rng.standard_normal((3, n))
    dv = rng.uniform(0, 100, n)
    rungs = ["a"] * 10 + ["b"] * 2
    rows = [
        {"arm": SLUG20, "eval_rung": "a", "rho_frozen": 0.0},
        {"arm": SLUG20, "eval_rung": "b", "rho_frozen": 0.0},
        {"arm": "arm7_map_ridge_pred", "eval_rung": "a", "rho_frozen": 0.0},
    ]
    frozen = {"arm7_map_ridge_pred": 2, SLUG20: 0}
    fair_score.attach_matched_layer_companion(
        rows, {SLUG20: sc}, dv, rungs, frozen, [5, 9, 13], arm=SLUG20, ref_arm="arm7_map_ridge_pred"
    )
    r_a, r_b, r_7 = rows
    assert r_a["matched_layer"] == 13 and r_a["matched_layer_idx"] == 2
    m = np.asarray([r == "a" for r in rungs])
    expected = float(arms.spearman_rows(sc[2][m][None], dv[m])[0])
    assert r_a["rho_matched_arm7_layer"] == expected
    assert r_a["n_eval_matched"] == 10
    assert "not a second method" in r_a["matched_note"]
    assert r_b["rho_matched_arm7_layer"] is None  # 2 rows < min_n
    assert r_b["n_eval_matched"] == 2
    assert "rho_matched_arm7_layer" not in r_7  # arm7's own row untouched


def test_matched_layer_fails_loud_without_reference_arm():
    """Companion-arm rows present but no reference frozen layer in scope ->
    SystemExit, never a silent own-argmax fallback — for BOTH registered
    pairs (arm20@arm7, arm8@arm12)."""
    import pytest

    for arm, ref in fair_score.MATCHED_COMPANIONS:
        rows = [{"arm": arm, "eval_rung": "a"}]
        sc = np.zeros((2, 4))
        with pytest.raises(SystemExit, match=f"{ref} has no frozen layer"):
            fair_score.attach_matched_layer_companion(
                rows, {arm: sc}, np.zeros(4), ["a"] * 4, {arm: 0}, [0, 1], arm=arm, ref_arm=ref
            )


def test_matched_layer_noop_without_companion_rows():
    """No companion-arm rows (skip cases) -> attach_matched_companions is a
    no-op even when the reference frozen layers are absent — the fail-loud
    fires only when there is a row the companion would otherwise miss."""
    rows = [{"arm": "arm7_map_ridge_pred", "eval_rung": "a", "rho_frozen": 0.1}]
    fair_score.attach_matched_companions(rows, {}, np.zeros(4), ["a"] * 4, {}, [0, 1])
    assert rows == [{"arm": "arm7_map_ridge_pred", "eval_rung": "a", "rho_frozen": 0.1}]


def test_matched_layer_on_real_evaluate_transfer_rows():
    """End-to-end at the real row grain: run_transfer_cell scores -> the REAL
    evaluate_transfer rows -> attach_matched_companions; the arm20 AND arm8
    rows carry their companions, each equal to the direct read off the scored
    profile at the reference arm's frozen layer."""
    data, cell, z_ev, dv_ev, za_ev = _toy_cell()
    scores, skipped = arms.run_transfer_cell(
        data,
        cell,
        z_ev,
        dv_ev,
        za_ev=za_ev,
        arms=list(fair_score.ROSTER_LINEAR),
        ridge_folds=(0,),
    )
    assert not skipped
    # arm7 + arm12 frozen at layer 1, everything else at 0 — so both
    # companions index a DIFFERENT layer than the arm's own frozen row.
    refs = {"arm7_map_ridge_pred", "arm12_oracle_reg"}
    frozen = {a: (1 if a in refs else 0) for a in scores}
    rungs = np.asarray(["r"] * len(dv_ev))
    rows, skips = arms.evaluate_transfer(
        scores,
        dv_ev,
        rungs,
        frozen,
        provenance={"mode": "fair-test"},
        cell=cell,
        layers=(0, 1),
        n_boot=8,
    )
    assert not skips
    fair_score.attach_matched_companions(rows, scores, dv_ev, rungs, frozen, [0, 1])
    for arm, field in ((SLUG20, "rho_matched_arm7_layer"), (SLUG8, "rho_matched_arm12_layer")):
        sub = [r for r in rows if r["arm"] == arm]
        assert len(sub) == 1, arm
        row = sub[0]
        assert row["matched_layer"] == 1 and row["matched_layer_idx"] == 1
        expected = float(arms.spearman_rows(scores[arm][1][None], np.asarray(dv_ev))[0])
        assert row[field] == expected, arm
    # reference arms' own rows carry no companion fields
    for ref in refs:
        r = next(x for x in rows if x["arm"] == ref)
        assert "rho_matched_arm7_layer" not in r and "rho_matched_arm12_layer" not in r


def test_collect_passes_matched_fields_and_compare_reports_gaps(monkeypatch, tmp_path):
    """The fig's collect() passes the matched-layer FIELDS through on the
    arm20 AND arm8 records (and only there), and compare()'s two matched
    checks report the same-layer gaps arm7_minus_arm20_matched /
    arm12_minus_arm8_matched — on a fully synthetic summary (no
    committed-artifact dependency)."""
    import json

    setting = fair_fig.SETTINGS["evil"][0]
    base = dict(ci_frozen=[0.3, 0.6], n_eval=100, map_kind="linear", eval_rung=setting)
    rows = [
        {**base, "arm": "arm4_ridge_ctx", "rho_frozen": 0.55, "layer": 18},
        {**base, "arm": "arm7_map_ridge_pred", "rho_frozen": 0.50, "layer": 20},
        {**base, "arm": "arm12_oracle_reg", "rho_frozen": 0.70, "layer": 17},
        {
            **base,
            "arm": "arm8_map_ridge_true",
            "rho_frozen": 0.52,
            "layer": 21,
            "rho_matched_arm12_layer": 0.47,
            "matched_layer": 17,
            "matched_layer_idx": 17,
            "n_eval_matched": 100,
            "matched_note": "confound check ... not a second method",
        },
        {
            **base,
            "arm": "arm20_shuffled_map_ridge",
            "rho_frozen": 0.48,
            "layer": 27,
            "rho_matched_arm7_layer": 0.41,
            "matched_layer": 20,
            "matched_layer_idx": 20,
            "n_eval_matched": 100,
            "matched_note": "confound check ... not a second method",
        },
    ]
    paths = {}
    for beh in fair_fig.BEHAVIORS:
        p = tmp_path / f"{beh}.json"
        p.write_text(json.dumps({"meta": {}, "transfer_rows": rows if beh == "evil" else []}))
        paths[beh] = p
    monkeypatch.setattr(fair_fig, "FAIR_SUMMARY", paths)
    monkeypatch.setattr(fair_fig, "ROOT", tmp_path)
    recs, _coverage, _meta = fair_fig.collect()
    r20 = [r for r in recs if r["method"] == "regression_shuffled_map"]
    r8 = [r for r in recs if r["method"] == "regression_realfit_mapped"]
    assert len(r20) == 1 and len(r8) == 1
    assert r20[0]["rho_matched_arm7_layer"] == 0.41 and r20[0]["matched_layer"] == 20
    assert r8[0]["rho_matched_arm12_layer"] == 0.47 and r8[0]["matched_layer"] == 17
    for r in (r20[0], r8[0]):
        assert "not a second method" in r["matched_note"]
    carriers = {id(r20[0]), id(r8[0])}
    assert not any(
        ("rho_matched_arm7_layer" in r or "rho_matched_arm12_layer" in r)
        for r in recs
        if id(r) not in carriers
    )
    # compare(): isolate the matched-check blocks from committed-artifact reads
    monkeypatch.setattr(fair_fig, "committed_v3", lambda: {})
    monkeypatch.setattr(fair_fig, "committed_arm_rows", lambda arm: {})
    monkeypatch.setattr(fair_fig, "committed_arm7_add", lambda: {})
    cmp = fair_fig.compare(recs)
    per_cell = cmp["arm20_matched_layer_check"]["per_cell"]
    assert len(per_cell) == 1
    c = per_cell[0]
    assert c["behavior"] == "evil" and c["setting"] == setting
    assert c["arm20_rho_own_argmax"] == 0.48 and c["arm20_own_layer"] == 27
    assert c["arm20_rho_at_arm7_layer"] == 0.41 and c["arm7_frozen_layer"] == 20
    assert c["arm7_rho"] == 0.50
    assert abs(c["arm7_minus_arm20_matched"] - (0.50 - 0.41)) < 1e-12
    per_cell8 = cmp["arm8_matched_layer_check"]["per_cell"]
    assert len(per_cell8) == 1
    c8 = per_cell8[0]
    assert c8["arm8_rho_own_frozen"] == 0.52 and c8["arm8_own_layer"] == 21
    assert c8["arm8_rho_at_arm12_layer"] == 0.47 and c8["arm12_frozen_layer"] == 17
    assert c8["arm12_rho"] == 0.70
    assert abs(c8["arm12_minus_arm8_matched"] - (0.70 - 0.47)) < 1e-12


def test_arm8_slot_bounded_and_label_unmistakable():
    """arm8's slot sits inside the CONTIGUOUS ceiling-bounded prefix (w·M(z)
    is a deterministic function of the context), its label leads with the
    fit-on-REAL / apply-to-MAPPED split (the arm's entire content), and
    arm7's label states its own fit+eval-on-mapped shape."""
    slot = fair_fig.METHOD_OF[(SLUG8, "linear")]
    assert slot == "regression_realfit_mapped"
    assert slot in fair_fig.CEILING_BOUNDED
    assert slot not in fair_fig.REAL_ANSWER_SLOTS
    labels = {m: lbl for _t, ms in fair_fig.GROUPS for m, lbl, _c, _h in ms}
    l8 = labels[slot].upper()
    assert "REAL" in l8 and "MAPPED" in l8 and "FIT" in l8
    l7 = labels["reg_map_linear"].lower()
    assert "fit" in l7 and "evaluated" in l7 and "mapped" in l7
