"""#1739 grid-fill: transfer roster widening, polarity split, preds schema.

Three deliverables are pinned here:

1. The WIDE transfer roster (the 6 core ladder arms + the fitted arms 5/7/8/12)
   and its rb-dependence partition, including the invariant that the registry's
   ``rb_dep`` flags agree with what ``run_cell_multi`` actually shares across
   regimes (a drifted flag would silently refit a shared arm per regime, or —
   worse — share an rb-DEPENDENT arm's scores across regimes).
2. The pvsynth polarity split (elicit / non_elicit / pooled) and its fail-loud
   group_key grammar.
3. The per-context transfer preds schema + the atomic JSONL writer.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore_persona_space.experiments.issue_1739 import arms  # noqa: E402
from explore_persona_space.experiments.issue_1739.fits import BudgetCell  # noqa: E402

WIDENED = ("arm5_mlp_ctx", "arm7_map_ridge_pred", "arm8_map_ridge_true", "arm12_oracle_reg")


# ---------------------------------------------------------------------------
# 1. roster + rb-dependence partition
# ---------------------------------------------------------------------------


def test_wide_roster_is_core_plus_the_four_fitted_arms():
    assert set(arms.TRANSFER_ARMS) < set(arms.TRANSFER_ARMS_WIDE)
    assert set(arms.TRANSFER_ARMS_WIDE) - set(arms.TRANSFER_ARMS) == set(WIDENED)
    # The core roster itself is untouched — the committed 6-arm columns are
    # reproducible byte-for-byte via `--arms core`.
    assert arms.TRANSFER_ARMS == (
        "arm1_ctx_e1",
        "arm3_identity_bias",
        "arm4_ridge_ctx",
        "arm6_map_proj_e1",
        "arm11_oracle_proj",
        "arm13_shuffled_map",
    )


def test_excluded_arms_stay_off_the_ladder():
    """9/14 (per-regime L2-SP), 10 (needs every fold), 15/16 (no eval text feats)."""
    for slug in (
        "arm9_pretrain_ft",
        "arm10_stacked",
        "arm14_shuffled_pt",
        "arm15_text_only",
        "arm16_surface_feat",
        "arm2_ctx_native",
    ):
        assert slug not in arms.TRANSFER_ARMS_WIDE


def test_resolve_transfer_roster_names_and_lists():
    assert arms.resolve_transfer_roster(None) == list(arms.TRANSFER_ARMS_WIDE)
    assert arms.resolve_transfer_roster("wide") == list(arms.TRANSFER_ARMS_WIDE)
    assert arms.resolve_transfer_roster("core") == list(arms.TRANSFER_ARMS)
    # nargs="+" hands a single-element list for `--transfer-arms wide`.
    assert arms.resolve_transfer_roster(["core"]) == list(arms.TRANSFER_ARMS)
    # Explicit list: deduped and returned in REGISTRY order, not caller order.
    got = arms.resolve_transfer_roster(["arm12_oracle_reg", "arm1_ctx_e1", "arm1_ctx_e1"])
    assert got == ["arm1_ctx_e1", "arm12_oracle_reg"]


def test_wide_nomlp_roster_is_wide_minus_the_mlp():
    """The staging roster: everything `wide` has except the expensive arm-5 MLP."""
    assert set(arms.TRANSFER_ARMS_WIDE) - set(arms.TRANSFER_ARMS_WIDE_NOMLP) == {"arm5_mlp_ctx"}
    assert arms.resolve_transfer_roster("wide-nomlp") == list(arms.TRANSFER_ARMS_WIDE_NOMLP)
    # It is NOT the default — the default stays the full wide roster.
    assert arms.resolve_transfer_roster(None) == list(arms.TRANSFER_ARMS_WIDE)
    # Its three added arms are the closed-form ridge ones (all rb-independent).
    added = set(arms.TRANSFER_ARMS_WIDE_NOMLP) - set(arms.TRANSFER_ARMS)
    assert added == {"arm7_map_ridge_pred", "arm8_map_ridge_true", "arm12_oracle_reg"}
    assert all(not arms.ARM_REGISTRY[a]["rb_dep"] for a in added)


def test_resolve_transfer_roster_rejects_unknown_never_silently_drops():
    with pytest.raises(ValueError, match="unknown arm slug"):
        arms.resolve_transfer_roster(["arm1_ctx_e1", "arm99_nope"])
    with pytest.raises(ValueError, match="unknown transfer roster"):
        arms.resolve_transfer_roster("everything")


def test_partition_transfer_roster_splits_wide_roster():
    rb_indep, rb_dep = arms.partition_transfer_roster(arms.TRANSFER_ARMS_WIDE)
    assert rb_indep == [
        "arm4_ridge_ctx",
        "arm5_mlp_ctx",
        "arm7_map_ridge_pred",
        "arm8_map_ridge_true",
        "arm12_oracle_reg",
    ]
    assert rb_dep == [
        "arm1_ctx_e1",
        "arm3_identity_bias",
        "arm6_map_proj_e1",
        "arm11_oracle_proj",
        "arm13_shuffled_map",
    ]
    assert set(rb_indep) | set(rb_dep) == set(arms.TRANSFER_ARMS_WIDE)
    assert not set(rb_indep) & set(rb_dep)
    # The pre-widening split had arm4 alone on the rb-independent side.
    assert arms.partition_transfer_roster(arms.TRANSFER_ARMS)[0] == ["arm4_ridge_ctx"]


def _toy_datas(n=24, d=5, ly=2, n_regimes=2, seed=0):
    """Two regime slices sharing every rb-independent input BY IDENTITY."""
    from explore_persona_space.experiments.issue_1739.fits import MapFit

    rng = np.random.default_rng(seed)
    z = rng.normal(size=(ly, n, d))
    za = z + 0.3 * rng.normal(size=(ly, n, d))
    dv = rng.normal(size=n)
    mapfit = MapFit(
        w=np.stack([np.eye(d) for _ in range(ly)]),
        x_mu=np.zeros((ly, 1, d)),
        x_sd=np.ones((ly, 1, d)),
        y_mu=np.zeros((ly, 1, d)),
        diagnostics={},
        kind="linear",
    )
    datas = [
        arms.CellData(
            z_ctx=z,
            z_ans=za,
            dv=dv,
            rb=rng.normal(size=(ly, d)),
            mapfit=mapfit,
            layers=tuple(range(ly)),
        )
        for _ in range(n_regimes)
    ]
    cell = BudgetCell(
        row_idx=np.arange(n),
        fold_ids=np.arange(n) % 3,
        n_folds=3,
        budget_l=n,
        draw=0,
        seed=0,
        fold_scheme="toy",
    )
    return datas, cell


def test_rb_dep_matches_dispatch():
    """The registry flag must agree with what run_cell_multi SHARES across regimes.

    ``_put_shared`` stores the SAME ndarray object in every regime's dict for
    rb-independent arms; rb-dependent arms get a distinct array per regime.
    Identity is therefore the ground truth for the flag, and this test reads it
    off a real (tiny) dispatch rather than restating the classification.
    """
    datas, cell = _toy_datas()
    outs = arms.run_cell_multi(datas, cell, arms=list(arms.TRANSFER_ARMS_WIDE), device="cpu")
    (s0, sk0), (s1, _sk1) = outs
    checked = 0
    for slug in arms.TRANSFER_ARMS_WIDE:
        if slug in sk0 or slug not in s0 or slug not in s1:
            continue  # arm skipped with a recorded reason — no identity to read
        checked += 1
        shared = s0[slug] is s1[slug]
        assert shared == (not arms.ARM_REGISTRY[slug]["rb_dep"]), (
            f"{slug}: run_cell_multi {'shares' if shared else 'does not share'} its scores "
            f"across regimes, but ARM_REGISTRY says rb_dep="
            f"{arms.ARM_REGISTRY[slug]['rb_dep']}"
        )
    assert checked >= len(arms.TRANSFER_ARMS_WIDE) - 1, f"only {checked} arms exercised"


def test_every_registry_arm_declares_rb_dep():
    for slug, spec in arms.ARM_REGISTRY.items():
        assert isinstance(spec.get("rb_dep"), bool), f"{slug} lacks a bool rb_dep"


def test_widened_arms_score_on_a_transfer_cell():
    """The four added arms actually produce eval-block scores (not silent skips)."""
    datas, cell = _toy_datas(n=24)
    z_ev = np.random.default_rng(7).normal(size=(2, 9, 5))
    za_ev = np.random.default_rng(8).normal(size=(2, 9, 5))
    dv_ev = np.random.default_rng(9).normal(size=9)
    scores, skipped = arms.run_transfer_cell(
        datas[0],
        cell,
        z_ev,
        dv_ev,
        za_ev=za_ev,
        arms=list(arms.TRANSFER_ARMS_WIDE),
        device="cpu",
        ridge_folds=(0,),
    )
    for slug in WIDENED:
        assert slug in scores, f"{slug} missing (skips: {skipped})"
        assert scores[slug].shape[1] == 9, (slug, scores[slug].shape)
        assert np.isfinite(scores[slug]).any(), f"{slug} produced all-NaN eval scores"
    assert not arms.roster_accounting_skips(arms.TRANSFER_ARMS_WIDE, scores, skipped)


def test_widened_arms_are_skipped_with_a_reason_when_answer_acts_are_absent():
    """No za -> arms 8/12 (and 11/3) record a reason; NEVER an unaccounted drop."""
    datas, cell = _toy_datas()
    bare = arms.CellData(
        z_ctx=datas[0].z_ctx,
        z_ans=None,
        dv=datas[0].dv,
        rb=datas[0].rb,
        mapfit=datas[0].mapfit,
        layers=datas[0].layers,
    )
    z_ev = np.random.default_rng(7).normal(size=(2, 9, 5))
    dv_ev = np.random.default_rng(9).normal(size=9)
    scores, skipped = arms.run_transfer_cell(
        bare,
        cell,
        z_ev,
        dv_ev,
        za_ev=None,
        arms=list(arms.TRANSFER_ARMS_WIDE),
        device="cpu",
        ridge_folds=(0,),
    )
    for slug in ("arm8_map_ridge_true", "arm12_oracle_reg", "arm11_oracle_proj"):
        assert slug in skipped and "answer activations" in skipped[slug]
    assert not arms.roster_accounting_skips(arms.TRANSFER_ARMS_WIDE, scores, skipped)


def test_roster_accounting_skips_flags_an_unaccounted_arm():
    recs = arms.roster_accounting_skips(
        ["arm1_ctx_e1", "arm5_mlp_ctx"], {"arm1_ctx_e1": np.zeros((1, 2))}, {}, variant="ctx"
    )
    assert [r["arm"] for r in recs] == ["arm5_mlp_ctx"]
    assert "roster-unaccounted" in recs[0]["reason"]
    assert recs[0]["variant"] == "ctx"


# ---------------------------------------------------------------------------
# 2. pvsynth polarity split
# ---------------------------------------------------------------------------


def _pvsynth_mod():
    import importlib

    return importlib.import_module("scripts.issue1739_pvsynth_arms")


def test_polarity_labels_parses_the_pvsynth_grammar():
    mod = _pvsynth_mod()
    keys = ["pvsynth-p0-pos", "pvsynth-p0-neg", "pvsynth-p4-pos"]
    assert mod.polarity_labels(keys, behavior="evil", variant="context_end") == [
        "pos",
        "neg",
        "pos",
    ]


def test_polarity_labels_fails_loud_on_an_unrecognized_group_key():
    mod = _pvsynth_mod()
    with pytest.raises(RuntimeError, match="group_key must end in"):
        mod.polarity_labels(
            ["pvsynth-p0-pos", "pvsynth-p1-middle"], behavior="evil", variant="context_end"
        )


def test_polarity_masks_partition_the_eval_block():
    """elicit + non_elicit tile pooled exactly — no context counted twice or lost."""
    mod = _pvsynth_mod()
    keys = [f"pvsynth-p{i // 2}-{'pos' if i % 2 == 0 else 'neg'}" for i in range(20)]
    pol = np.asarray(mod.polarity_labels(keys, behavior="evil", variant="context_end"))
    elicit, non_elicit = pol == "pos", pol == "neg"
    assert int(elicit.sum()) == int(non_elicit.sum()) == 10
    assert not (elicit & non_elicit).any()
    assert (elicit | non_elicit).all()


def test_polarity_subset_rows_use_the_same_evaluate_transfer_semantics():
    """A masked-slice subset row equals the row from evaluating that slice directly."""
    rng = np.random.default_rng(3)
    n = 20
    scores = {"arm1_ctx_e1": rng.normal(size=(2, n))}
    dv = rng.normal(size=n)
    rungs = np.asarray(["pvsynth"] * n)
    pol = np.asarray(["pos" if i % 2 == 0 else "neg" for i in range(n)])
    cell = BudgetCell(np.arange(4), np.arange(4) % 2, 2, 4, 0, 0, "toy")
    mask = pol == "pos"
    sub, _ = arms.evaluate_transfer(
        {k: v[:, mask] for k, v in scores.items()},
        dv[mask],
        rungs[mask],
        {"arm1_ctx_e1": 0},
        provenance={"polarity_subset": "elicit"},
        cell=cell,
        n_boot=32,
    )
    direct, _ = arms.evaluate_transfer(
        {k: v[:, mask] for k, v in scores.items()},
        dv[mask],
        rungs[mask],
        {"arm1_ctx_e1": 0},
        provenance={},
        cell=cell,
        n_boot=32,
    )
    assert len(sub) == 1
    assert sub[0]["polarity_subset"] == "elicit"
    assert sub[0]["n_eval"] == 10
    assert sub[0]["rho_frozen"] == direct[0]["rho_frozen"]
    assert sub[0]["ci_frozen"] == direct[0]["ci_frozen"]


# ---------------------------------------------------------------------------
# 3. per-context transfer preds
# ---------------------------------------------------------------------------


def test_transfer_preds_rows_schema_and_labels():
    scores = {"arm1_ctx_e1": np.array([[0.0, 1.0, 2.0], [9.0, 9.0, 9.0]])}
    rows = arms.transfer_preds_rows(
        scores,
        np.array([10.0, 20.0, 30.0]),
        ["c0", "c1", "c2"],
        {"arm1_ctx_e1": 0},
        provenance={"behavior": "evil", "variant": "context_end"},
        layers=(4, 7),
        labels={"polarity": ["pos", "neg", "pos"]},
    )
    assert len(rows) == 3
    assert [r["score"] for r in rows] == [0.0, 1.0, 2.0]  # frozen layer 0, not layer 1
    assert [r["context_id"] for r in rows] == ["c0", "c1", "c2"]
    assert [r["polarity"] for r in rows] == ["pos", "neg", "pos"]
    assert rows[0]["layer"] == 4 and rows[0]["frozen_layer_idx"] == 0
    assert rows[0]["behavior"] == "evil"


def test_transfer_preds_rows_skips_arms_without_a_frozen_layer():
    scores = {"a": np.zeros((1, 2)), "b": np.zeros((1, 2))}
    rows = arms.transfer_preds_rows(
        scores, np.zeros(2), ["c0", "c1"], {"a": 0}, provenance={}, layers=()
    )
    assert {r["arm"] for r in rows} == {"a"}


def test_transfer_preds_rows_rejects_misaligned_labels():
    with pytest.raises(ValueError, match="label column"):
        arms.transfer_preds_rows(
            {"a": np.zeros((1, 2))},
            np.zeros(2),
            ["c0", "c1"],
            {"a": 0},
            provenance={},
            labels={"polarity": ["pos"]},
        )


def test_write_preds_jsonl_is_atomic_and_truncating(tmp_path):
    p = tmp_path / "sub" / "preds.jsonl"
    arms.write_preds_jsonl(p, [{"arm": "a", "score": 1.0}, {"arm": "b", "score": 2.0}])
    assert [json.loads(x)["arm"] for x in p.read_text().split("\n") if x.strip()] == ["a", "b"]
    arms.write_preds_jsonl(p, [{"arm": "c", "score": 3.0}])  # re-run OVERWRITES, never appends
    assert [json.loads(x)["arm"] for x in p.read_text().split("\n") if x.strip()] == ["c"]
    assert not list(p.parent.glob("*.tmp"))


# ---------------------------------------------------------------------------
# CLI binds (the deferred-import / signature classes that only fire in prod)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script",
    [
        "issue1739_fits.py",
        "issue1739_wcrung_arms.py",
        "issue1739_pvsynth_arms.py",
        "issue1739_stage_percell_preds.py",
    ],
)
def test_script_help_parses(script):
    """--help exercises the real parser: a malformed add_argument fails here."""
    r = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / script), "--help"],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=REPO_ROOT,
    )
    assert r.returncode == 0, r.stderr[-2000:]


def test_fits_parser_accepts_transfer_arms_and_resolves_wide():
    import importlib

    mod = importlib.import_module("scripts.issue1739_fits")
    args = mod._parse_args(["--transfer-arms", "core"])
    assert arms.resolve_transfer_roster(args.transfer_arms) == list(arms.TRANSFER_ARMS)
    args = mod._parse_args([])
    assert arms.resolve_transfer_roster(args.transfer_arms) == list(arms.TRANSFER_ARMS_WIDE)


def test_stager_recorded_names_reads_jsonl_without_splitlines(tmp_path):
    """U+2028 inside a JSON string must not shred the record (gotchas.md)."""
    mod = __import__("scripts.issue1739_stage_percell_preds", fromlist=["x"])
    p = tmp_path / "cells.jsonl"
    # chr(0x2028) built from its code point, never a source literal: a raw
    # U+2028 in source trips ruff RUF001, and a backslash-u escape typed
    # through the Edit tool is silently decoded to the literal before it lands.
    unit_key = "a" + chr(0x2028) + "b"
    p.write_text(
        json.dumps({"unit_key": unit_key, "preds_npz": "aa.npz"}, ensure_ascii=False)
        + "\n"
        + json.dumps({"unit_key": "c", "preds_npz": "bb.npz"})
        + "\n",
        encoding="utf-8",
    )
    # str.splitlines() splits ON the U+2028 and would shred the first record.
    assert len(p.read_text(encoding="utf-8").splitlines()) == 3
    assert mod.recorded_preds_names(p) == {"aa.npz", "bb.npz"}


def test_stager_fails_loud_when_no_cell_records_predictions(tmp_path):
    mod = __import__("scripts.issue1739_stage_percell_preds", fromlist=["x"])
    p = tmp_path / "cells.jsonl"
    p.write_text(json.dumps({"unit_key": "a"}) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="a re-score IS required"):
        mod.recorded_preds_names(p)


# ---------------------------------------------------------------------------
# 4. new-arm-round item 2: nonlinear oracle arms 17/18
# ---------------------------------------------------------------------------

ORACLE_ROSTER = ("arm12_oracle_reg", "arm17_oracle_mlp", "arm18_oracle_krr")


def test_oracle_arms_registered_rb_independent_and_off_committed_rosters():
    for slug in ("arm17_oracle_mlp", "arm18_oracle_krr"):
        spec = arms.ARM_REGISTRY[slug]
        assert spec["family"] == "oracle" and spec["layered"] is True
        assert spec["rb_dep"] is False
        # The committed named rosters are UNTOUCHED — arm17/18 run only via an
        # explicit slug list (the oracle-leg dispatch).
        assert slug not in arms.TRANSFER_ARMS
        assert slug not in arms.TRANSFER_ARMS_WIDE
        assert slug not in arms.TRANSFER_ARMS_WIDE_NOMLP
    rb_indep, rb_dep = arms.partition_transfer_roster(ORACLE_ROSTER)
    assert rb_indep == list(ORACLE_ROSTER) and rb_dep == []
    assert arms.resolve_transfer_roster(list(ORACLE_ROSTER)) == list(ORACLE_ROSTER)


def test_oracle_arms_share_scores_across_regimes_and_score_transfer():
    """REAL dispatch body: arm17 (batched MLP on za) + arm18 (Nystrom KRR on
    za) produce finite scores shared BY IDENTITY across regime slices
    (rb_dep=False ground truth, same read as test_rb_dep_matches_dispatch)
    and score the eval block on a transfer cell under ridge_folds=(0,)."""
    datas, cell = _toy_datas()
    outs = arms.run_cell_multi(
        datas,
        cell,
        arms=list(ORACLE_ROSTER),
        device="cpu",
        mlp_kwargs={"max_epochs": 3, "hidden": 8},
    )
    (s0, sk0), (s1, _sk1) = outs
    assert not sk0, sk0
    for slug in ORACLE_ROSTER:
        assert s0[slug] is s1[slug], f"{slug} not shared across regimes"
        assert s0[slug].shape == (2, 24), (slug, s0[slug].shape)
        assert np.isfinite(s0[slug]).all(), slug
    rng = np.random.default_rng(7)
    z_ev = rng.normal(size=(2, 9, 5))
    za_ev = rng.normal(size=(2, 9, 5))
    dv_ev = rng.normal(size=9)
    scores, skipped = arms.run_transfer_cell(
        datas[0],
        cell,
        z_ev,
        dv_ev,
        za_ev=za_ev,
        arms=list(ORACLE_ROSTER),
        device="cpu",
        ridge_folds=(0,),
    )
    assert not skipped, skipped
    for slug in ORACLE_ROSTER:
        assert scores[slug].shape[1] == 9, (slug, scores[slug].shape)
        assert np.isfinite(scores[slug]).any(), f"{slug} produced all-NaN eval scores"
    assert not arms.roster_accounting_skips(ORACLE_ROSTER, scores, skipped)


def test_oracle_arms_skip_with_reason_without_answer_acts():
    datas, cell = _toy_datas()
    bare = arms.CellData(
        z_ctx=datas[0].z_ctx,
        z_ans=None,
        dv=datas[0].dv,
        rb=datas[0].rb,
        mapfit=datas[0].mapfit,
        layers=datas[0].layers,
    )
    scores, skipped = arms.run_cell(
        bare, cell, arms=["arm17_oracle_mlp", "arm18_oracle_krr"], device="cpu"
    )
    for slug in ("arm17_oracle_mlp", "arm18_oracle_krr"):
        assert slug not in scores
        assert skipped[slug] == "no answer activations"


def test_rowset_seed_is_order_invariant_and_value_sensitive():
    """arm18's inner split is a pure function of the fold's TRAIN ROW SET, so
    the transfer leg's rb-independent row-set cache stays exact."""
    a = arms._rowset_seed(np.array([3, 1, 2]))
    assert a == arms._rowset_seed(np.array([1, 2, 3]))
    assert a != arms._rowset_seed(np.array([1, 2, 4]))
    assert 0 <= a < 2**31
