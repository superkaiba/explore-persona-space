"""End-to-end tests for the #1481 Phase B/D analysis + figures drivers.

Fixtures are SCHEMA-REAL synthetic inputs shaped exactly like the round-1/2A
smoke outputs (fu4 ladders `runs[<id>].{rates_by_step,selection}`, organisms
`completions__{side}__{ctx}.json`, marker `selection.json` / `ladder.json` /
`panel/rung*.json` / `slot_reads_rung*.json` four-float reads, mix_meta.json).
The ONLY fake is the smoke-only stub judge at the external API boundary
(signature-conformant by construction — it mirrors the organisms JudgeFn
seam); every production body (`_judge_cell`, `phase_select`, `phase_judge`,
`phase_contrast`, the MF-3 assert, the figure families) executes for real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1090_fu3_worker as fu3w  # noqa: E402
import issue1090_fu4 as fu4  # noqa: E402
import issue1481_analysis as ana  # noqa: E402
import issue1481_cells as cells  # noqa: E402
import issue1481_marker as mk  # noqa: E402
import issue1481_worker as wk  # noqa: E402

from explore_persona_space.artifacts.organisms import _sha256_text  # noqa: E402

QUESTIONS = ["How do I sort a list?", "What is a decorator?", "Explain recursion."]
QUESTIONS_SHA = _sha256_text(json.dumps(list(QUESTIONS), ensure_ascii=False))
BEH = "impolite"
BEH_KEY = "imp"
CTXS = ("pers", "bare")
PANEL_IDS = [c.context_id for c in fu3w.bystander_panel(BEH)]
# Plan-§5 realized training panels (5-member; bare excludes its own source).
REALIZED_PERS = ["default", "neg_sp_police", "neg_sp_ph4", "neg_extra_a", "neg_extra_b"]
REALIZED_BARE = ["neg_sp_police", "neg_sp_ph4", "neg_extra_a", "neg_extra_b"]


def _sel(step: int, rate: float, in_band: bool) -> dict:
    return {"step": step, "rate": rate, "in_band": in_band, "fallback": None}


def _run_rec(rates: dict[int, float], sel_step: int) -> dict:
    lo, hi = cells.JUDGED_RATE_BAND
    rate = rates[sel_step]
    return {
        "rates_by_step": {str(k): v for k, v in sorted(rates.items())},
        "selection": _sel(sel_step, rate, lo <= rate <= hi),
    }


def build_content_fixtures(root: Path) -> dict[str, Path]:
    """Fresh i1481 round ladders + the reused fu4/fu5 committed parents."""
    ladders_dir = root / "ladders"
    repo_root = root / "repo"
    tags = {lr: fu4.LR_TAG[lr] for lr in fu4.FU4_LRS}
    # Per-LR ladder shapes: lowest lr in band -> the deterministic verdict arm.
    con_rates = {
        tags[1e-5]: ({10: 0.20, 20: 0.65, 30: 0.80}, 20),
        tags[3e-5]: ({10: 0.70, 20: 0.90, 30: 0.95}, 10),
        tags[1e-4]: ({10: 0.90, 20: 0.95, 30: 0.99}, 10),
    }
    po_rates = {
        tags[1e-5]: ({10: 0.30, 20: 0.71, 30: 0.66}, 20),
        tags[3e-5]: ({10: 0.72, 20: 0.93, 30: 0.97}, 10),
        tags[1e-4]: ({10: 0.92, 20: 0.97, 30: 0.99}, 10),
    }
    for regime, shapes in (("con", con_rates), ("po", po_rates)):
        runs: dict[str, dict] = {}
        for ctx in CTXS:
            for lr in fu4.FU4_LRS:
                for seed in cells.SEEDS:
                    if cells.is_reused(BEH_KEY, ctx, regime, lr, seed):
                        continue
                    arm_id = f"{BEH_KEY}-{ctx}-{regime}-{tags[lr]}-s{seed}"
                    rates, sel_step = shapes[tags[lr]]
                    runs[arm_id] = _run_rec(rates, sel_step)
        path = ladders_dir / f"{cells.round_name(BEH_KEY, regime)}_ladders.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"runs": runs}))
    # Reused parent ladders (imp-pers/imp-conv via fu4, imp-bare via fu5).
    for rel, src_ids in (
        (
            "eval_results/issue_1090/fu4-extended-dose-lr/fu4_ladders.json",
            [f"imp-pers-{t}" for t in tags.values()] + [f"imp-conv-{t}" for t in tags.values()],
        ),
        (
            "eval_results/issue_1090/finish-impolite-bare-and-formatting-rank/fu5_ladders.json",
            [f"imp-bare-{t}" for t in tags.values()],
        ),
    ):
        runs = {}
        for src in src_ids:
            tag = src.rsplit("-", 1)[1]
            rates, sel_step = con_rates[tag]
            runs[src] = _run_rec(rates, sel_step)
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"runs": runs}))
    return {"ladders_dir": ladders_dir, "repo_root": repo_root}


def _completions_file(path: Path, n: int = 2) -> None:
    payload = {
        "manifest": {"questions_sha256": QUESTIONS_SHA},
        "completions": [
            [f"completion {qi}-{ci}" for ci in range(n)] for qi in range(len(QUESTIONS))
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def build_panel_fixtures(root: Path, manifest: dict) -> dict[str, Path]:
    panel_root = root / "panel"
    base_root = root / "base"
    for ctx_id in PANEL_IDS:
        _completions_file(base_root / f"completions__base__{ctx_id}.json")
    verdict_arms = set()
    for cell in manifest["content"][BEH_KEY].values():
        for srec in cell["seeds"].values():
            for regime in ("con", "po"):
                verdict_arms.add(srec[regime]["arm_id"])
    for arm_id in sorted(verdict_arms):
        for ctx_id in PANEL_IDS:
            _completions_file(panel_root / arm_id / f"completions__trained__{ctx_id}.json")
    return {"panel_root": panel_root, "base_root": base_root}


def build_mix_meta_fixtures(root: Path) -> Path:
    mix_root = root / "mix_metas"
    for ctx, realized in (("pers", REALIZED_PERS), ("bare", REALIZED_BARE)):
        path = mix_root / f"{BEH_KEY}-{ctx}-con" / "mix_meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "counts_realized": {"positives": 20, "negatives": 40, "generic": 40},
                    "panel_context_ids": realized,
                }
            )
        )
    return mix_root


def _marker_probe_row(ctx_id: str, qi: int, trained: dict, base: dict) -> dict:
    return {"row": {"context_id": ctx_id, "q": qi}, "trained": trained, "base": base}


def build_marker_fixtures(root: Path) -> Path:
    """All 48 marker runs, schema-real per the marker dispatcher's writers:
    selection.json (incl. ``ceiling_rung`` + ``panel_rungs`` battery roles),
    ladder.json; panel batteries at TWO rungs — the emission-onset rung 10
    (half-dose: source ΔG/2, non-source ΔG/4, margin fraction 0.75) and the
    selected+ceiling rung 20 (source ΔG con=6 / po=7, uniform margins) — and
    slot reads at the selected rung, all with four-float per-probe rows."""
    marker_root = root / "marker"
    base = {"logp": -20.0, "z_marker": -4.0, "z_eos": 6.0, "logZ": 10.0}
    for ctx_key in mk.CTX_KEYS:
        src_id = mk.CTX_SOURCE_ID[ctx_key]
        panel_ctx_ids = [src_id] + [f"mkread_{ctx_key}_{i}" for i in range(5)]
        for regime in ("con", "po"):
            base_logp = -20.0
            trained_logp = -14.0 if regime == "con" else -13.0
            for lr_key in mk.LR_ARMS:
                for seed in mk.SEEDS:
                    run_id = mk.run_id_for(ctx_key, regime, lr_key, seed)
                    d = marker_root / run_id
                    d.mkdir(parents=True, exist_ok=True)
                    step = 20
                    dg = trained_logp - base_logp
                    (d / "selection.json").write_text(
                        json.dumps(
                            {
                                "step": step,
                                "in_window": True,
                                "fallback": None,
                                "delta_logp_mean": dg,
                                "window": [5.0, 12.0],
                                "run_id": run_id,
                                "ctx_key": ctx_key,
                                "regime": regime,
                                "lr_key": lr_key,
                                "seed": seed,
                                "emission_onset_rung": 10,
                                "ceiling_rung": 20,
                                "selectivity_break_rung": None,
                                "selectivity_break_resolution": [10, 20],
                                "panel_rungs": {
                                    "10": ["emission_onset"],
                                    "20": ["ceiling", "selected"],
                                },
                            }
                        )
                    )
                    (d / "ladder.json").write_text(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "reads_by_step": {
                                    "10": {
                                        "delta_logp_mean": dg / 2,
                                        "delta_margin_mean": dg / 2,
                                        "source_emission_rate": 0.0,
                                        "gen_emission_rate": 0.05,
                                    },
                                    "20": {
                                        "delta_logp_mean": dg,
                                        "delta_margin_mean": dg,
                                        "source_emission_rate": 0.0,
                                        "gen_emission_rate": 0.1,
                                    },
                                },
                            }
                        )
                    )
                    # Selected+ceiling rung 20: uniform ΔG + margins (Δmargin
                    # (2-5)-(-4-6) = 7 everywhere → transfer fraction 1.0).
                    trained20 = {"logp": trained_logp, "z_marker": 2.0, "z_eos": 5.0, "logZ": 10.0}
                    per_probe = [
                        _marker_probe_row(ctx_id, qi, trained20, base)
                        for ctx_id in panel_ctx_ids
                        for qi in range(len(QUESTIONS))
                    ]
                    # Emission-onset rung 10: half-dose — source ΔG/2 with
                    # Δmargin (-1-5)-(-10) = 4; non-source ΔG/4 with Δmargin
                    # (-2-5)-(-10) = 3 → margin transfer fraction 3/4 = 0.75.
                    src10 = {
                        "logp": base_logp + dg / 2,
                        "z_marker": -1.0,
                        "z_eos": 5.0,
                        "logZ": 10.0,
                    }
                    ns10 = {
                        "logp": base_logp + dg / 4,
                        "z_marker": -2.0,
                        "z_eos": 5.0,
                        "logZ": 10.0,
                    }
                    per_probe10 = [
                        _marker_probe_row(ctx_id, qi, src10 if ctx_id == src_id else ns10, base)
                        for ctx_id in panel_ctx_ids
                        for qi in range(len(QUESTIONS))
                    ]
                    (d / "panel").mkdir(exist_ok=True)
                    (d / "panel" / "rung10.json").write_text(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "step": 10,
                                "roles": ["emission_onset"],
                                "source_context": src_id,
                                "per_probe": per_probe10,
                            }
                        )
                    )
                    (d / "panel" / f"rung{step}.json").write_text(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "step": step,
                                "roles": ["candidate", "ceiling", "selected"],
                                "source_context": src_id,
                                "per_probe": per_probe,
                            }
                        )
                    )
                    (d / f"slot_reads_rung{step}.json").write_text(
                        json.dumps({"per_probe": per_probe})
                    )
    return marker_root


def build_margin_fixtures(root: Path, manifest: dict) -> Path:
    content_root = root / "content_root"
    for i, (arm_id, _arm) in enumerate(
        sorted(
            (a, r)
            for cell in manifest["content"][BEH_KEY].values()
            for a, r in cell["arms"].items()
        )
    ):
        path = content_root / arm_id / "margin.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"margin": 0.1 + 0.02 * i}))
    return content_root


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> dict:
    """One full select → judge(stub) → contrast run over the fixtures."""
    root = tmp_path_factory.mktemp("i1481")
    paths = build_content_fixtures(root)
    marker_root = build_marker_fixtures(root)
    out_dir = root / "analysis"
    rc = ana.main(
        [
            "select",
            "--out-dir",
            str(out_dir),
            "--ladders-dir",
            str(paths["ladders_dir"]),
            "--repo-root",
            str(paths["repo_root"]),
            "--marker-root",
            str(marker_root),
            "--behaviors",
            "imp",
            "--contexts",
            "pers,bare",
        ]
    )
    assert rc == 0
    manifest = json.loads((out_dir / "verdict_manifest.json").read_text())
    gen = build_panel_fixtures(root, manifest)
    qpath = root / "questions.json"
    qpath.write_text(json.dumps(QUESTIONS))
    rc = ana.main(
        [
            "judge",
            "--out-dir",
            str(out_dir),
            "--behavior",
            BEH,
            "--panel-root",
            str(gen["panel_root"]),
            "--base-panel-root",
            str(gen["base_root"]),
            "--questions",
            str(qpath),
            "--smoke",
            "--stub-judge",
        ]
    )
    assert rc == 0
    mix_root = build_mix_meta_fixtures(root)
    content_root = build_margin_fixtures(root, manifest)
    rc = ana.main(
        [
            "contrast",
            "--out-dir",
            str(out_dir),
            "--manifest",
            str(out_dir / "verdict_manifest.json"),
            "--aggregates-dir",
            str(out_dir),
            "--mix-meta-root",
            str(mix_root),
            "--marker-root",
            str(marker_root),
            "--content-root",
            str(content_root),
        ]
    )
    assert rc == 0
    return {
        "root": root,
        "out_dir": out_dir,
        "manifest": manifest,
        "mix_root": mix_root,
        "marker_root": marker_root,
        "gen": gen,
        "qpath": qpath,
        "content_root": content_root,
        "paths": paths,
    }


def test_select_verdict_arms_and_dose(pipeline):
    manifest = pipeline["manifest"]
    tag = fu4.LR_TAG[1e-5]
    for ctx in CTXS:
        for seed in cells.SEEDS:
            srec = manifest["content"][BEH_KEY][ctx]["seeds"][str(seed)]
            # Lowest-LR in-band arm wins the verdict (#1434 verbatim rule).
            assert srec["con"]["rule"] == "lowest_lr_in_band"
            assert srec["con"]["arm_id"].endswith(f"-{tag}-s{seed}")
            assert srec["po"]["arm_id"].endswith(f"-{tag}-s{seed}")
            dose = srec["dose"]
            assert dose["dose_matched"] is True
            assert dose["rate_gap"] == pytest.approx(0.06)
            nd = srec["nearest_dose_po"]
            assert nd["rate"] == pytest.approx(0.66)
            assert nd["differs_from_verdict"] is True
    dispatch = manifest["panel_dispatch"]
    # Seed-42 con verdict arms are the reused fu4/fu5 committed checkpoints.
    assert f"imp-pers-con-{tag}-s42" in dispatch["reused_ckpt_arms"]
    assert f"imp-bare-con-{tag}-s42" in dispatch["reused_ckpt_arms"]
    assert f"imp-pers-con-{tag}-s137" in dispatch["fresh_arms"]


def test_judge_aggregate_shape(pipeline):
    agg = json.loads((pipeline["out_dir"] / "panel_aggregate_imp.json").read_text())
    assert agg["smoke_stub_judge"] is True
    assert agg["instrument"] == "stub-smoke"
    assert set(agg["base_panel"]) == set(PANEL_IDS)
    assert len(agg["arms"]) == 8  # 2 ctx x 2 regimes x 2 seeds verdict arms
    rec = next(iter(agg["arms"].values()))["contexts"][PANEL_IDS[0]]
    assert rec["n_items"] == len(QUESTIONS) * 2
    assert rec["n_scored"] == rec["n_items"]
    assert 0.0 <= rec["rate"] <= 1.0
    assert rec["wilson_95"][0] <= rec["rate"] <= rec["wilson_95"][1]


def test_contrast_content_blocks(pipeline):
    content = json.loads((pipeline["out_dir"] / "regime_contrast_content.json").read_text())
    assert content["smoke_stub_judge"] is True
    for ctx in CTXS:
        cell = content["behavior_contexts"][BEH_KEY][ctx]
        assert cell["pooled"]["status"] == "computed"
        assert cell["pooled"]["lattice"] in ("Containment", "Reversed", "Indistinguishable")
        lo, hi = cell["pooled"]["newcombe_95"]
        assert lo <= cell["pooled"]["D"] <= hi
        assert len(cell["per_seed_Ds"]) == 2
        assert cell["heldout"]["mf3_checked"] is True
        assert cell["heldout"]["pooled"]["status"] == "computed"
        assert set(cell["heldout"]["contexts"]) == set(ana.registered_heldout(BEH, ctx))
        assert len(cell["per_context"]) == 5  # six-context panel minus the source
        for blk in cell["per_context"]:
            assert blk["status"] == "computed"
            assert "wilson_95" in blk["po"]
    headline = content["behavior_headline"][BEH_KEY]
    assert headline["realized_dose_matched_denominator"] == 2
    assert headline["status"] == "computed"


def test_contrast_marker_bootstrap(pipeline):
    marker = json.loads((pipeline["out_dir"] / "regime_contrast_marker.json").read_text())
    assert marker["bootstrap"] == {"draws": 2000, "seed": 653, "cluster": "question"}
    for ctx_key in mk.CTX_KEYS:
        cell = marker["contexts"][ctx_key]
        pooled = cell["pooled_nonsource"]
        # Fixture pairs po ΔG=7 vs con ΔG=6 on every (ctx, q) -> D = +1 nat,
        # a degenerate bootstrap CI at [1, 1], lattice Containment.
        assert pooled["D_nats"] == pytest.approx(1.0)
        assert pooled["bootstrap_95"] == pytest.approx([1.0, 1.0])
        assert pooled["lattice"] == "Containment"
        assert cell["dose_matched"] is True
    assert marker["three_space"]
    rec = next(iter(marker["three_space"].values()))
    assert rec["delta_z_marker_mean"] == pytest.approx(6.0)
    assert rec["divergence_points"]


def test_contrast_marker_dose_curves(pipeline):
    """Plan §6 install-strength read 3 (concern marker-dose-curves-analysis):
    leakage-vs-install dose curves at the panel rungs per marker cell — ΔG +
    EOS-margin transfer fractions (never raw-log-P fractions) — plus the full
    source-install trajectory from the per-rung ladder."""
    marker = json.loads((pipeline["out_dir"] / "regime_contrast_marker.json").read_text())
    curves = marker["dose_curves"]
    assert len(curves) == len(mk.CTX_KEYS) * 2 * len(mk.LR_ARMS) * len(mk.SEEDS)
    lr_key = next(iter(mk.LR_ARMS))
    ctx_key, seed = mk.CTX_KEYS[0], mk.SEEDS[0]
    for regime, dg in (("con", 6.0), ("po", 7.0)):
        cell = curves[mk.run_id_for(ctx_key, regime, lr_key, seed)]
        assert cell["regime"] == regime
        assert cell["selected_step"] == 20
        assert [r["step"] for r in cell["rungs"]] == [10, 20]
        r10, r20 = cell["rungs"]
        assert r10["roles"] == ["emission_onset"]
        assert r20["roles"] == ["ceiling", "selected"]
        # Source install (the x-axis) in BOTH spaces per rung.
        assert r20["source_install_logp"] == pytest.approx(dg)
        assert r20["source_install_margin"] == pytest.approx(7.0)
        assert r10["source_install_logp"] == pytest.approx(dg / 2)
        assert r10["source_install_margin"] == pytest.approx(4.0)
        # Leakage ΔG + the EOS-margin transfer fraction (margin space only).
        assert r20["nonsource_delta_logp_mean"] == pytest.approx(dg)
        assert r20["nonsource_margin_transfer_fraction_mean"] == pytest.approx(1.0)
        assert r10["nonsource_delta_logp_mean"] == pytest.approx(dg / 4)
        assert r10["nonsource_margin_transfer_fraction_mean"] == pytest.approx(0.75)
        # Per-context blocks: 6 contexts, source flagged, source fraction None.
        assert len(r20["per_context"]) == 6
        src = cell["source_context"]
        assert r20["per_context"][src]["is_source"] is True
        assert r20["per_context"][src]["margin_transfer_fraction"] is None
        # Raw-alongside-processed companion rows: 5 non-source ctx x questions.
        assert len(r10["per_question"]) == 5 * len(QUESTIONS)
        assert {p["context_id"] for p in r10["per_question"]} == set(r10["per_context"]) - {src}
        # Full source-install trajectory copied from the ladder.
        assert [t["step"] for t in cell["trajectory"]] == [10, 20]
        assert cell["trajectory"][1]["delta_logp_mean"] == pytest.approx(dg)
        assert cell["trajectory"][0]["gen_emission_rate"] == pytest.approx(0.05)


def test_dose_curves_missing_panel_read_fails_loud(pipeline, tmp_path):
    """A battery panel file missing for a panel_rungs entry is a RuntimeError
    (fail-fast), never a silently skipped rung."""
    import shutil

    marker_root = tmp_path / "marker"
    shutil.copytree(pipeline["marker_root"], marker_root)
    manifest = pipeline["manifest"]
    victim = next(iter(manifest["marker"]["arms"]))
    (marker_root / victim / "panel" / "rung10.json").unlink()
    with pytest.raises(RuntimeError, match="panel read missing"):
        ana._marker_dose_curves(manifest, marker_root)


def test_margin_rate_validation(pipeline):
    validation = json.loads((pipeline["out_dir"] / "margin_rate_validation.json").read_text())
    rec = validation[BEH_KEY]
    assert rec["n_pairs"] >= 3
    assert "spearman_rho" in rec


def test_mf3_violation_raises(pipeline, tmp_path):
    """Designed violation probe: a realized panel that CONTAINS a registered
    held-out context must fail loud BEFORE any held-out D is computed."""
    bad_root = tmp_path / "mix_metas_bad"
    for ctx, realized in (("pers", REALIZED_PERS), ("bare", REALIZED_BARE)):
        path = bad_root / f"{BEH_KEY}-{ctx}-con" / "mix_meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        violating = [*realized, "wildchat_prefix_real545"]
        path.write_text(json.dumps({"panel_context_ids": violating}))
    out_dir = pipeline["out_dir"]
    with pytest.raises(RuntimeError, match=r"i1481-MF3.*held-out set .* REALIZED"):
        ana.main(
            [
                "contrast",
                "--out-dir",
                str(tmp_path / "out"),
                "--manifest",
                str(out_dir / "verdict_manifest.json"),
                "--aggregates-dir",
                str(out_dir),
                "--mix-meta-root",
                str(bad_root),
            ]
        )


def test_mf3_derived_vs_registered_drift_raises(pipeline, tmp_path):
    """A realized panel that silently re-scopes the derived held-out set
    (without intersecting it) is ALSO a fail-loud, not a silent re-scope."""
    bad_root = tmp_path / "mix_metas_drift"
    for ctx in CTXS:
        path = bad_root / f"{BEH_KEY}-{ctx}-con" / "mix_meta.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        # Dropping neg_sp_police from the realized panel leaves it in the
        # DERIVED held-out set -> derived != registered.
        path.write_text(json.dumps({"panel_context_ids": ["neg_sp_ph4", "neg_extra_a"]}))
    out_dir = pipeline["out_dir"]
    with pytest.raises(RuntimeError, match=r"i1481-MF3.*derived held-out set"):
        ana.main(
            [
                "contrast",
                "--out-dir",
                str(tmp_path / "out"),
                "--manifest",
                str(out_dir / "verdict_manifest.json"),
                "--aggregates-dir",
                str(out_dir),
                "--mix-meta-root",
                str(bad_root),
            ]
        )


def test_stub_judge_requires_smoke(pipeline, tmp_path):
    with pytest.raises(SystemExit, match="smoke-only"):
        ana.main(
            [
                "judge",
                "--out-dir",
                str(tmp_path / "o"),
                "--behavior",
                BEH,
                "--panel-root",
                str(pipeline["gen"]["panel_root"]),
                "--base-panel-root",
                str(pipeline["gen"]["base_root"]),
                "--stub-judge",
            ]
        )


def test_judge_questions_mismatch_fails_loud(pipeline, tmp_path):
    qpath = tmp_path / "other_questions.json"
    qpath.write_text(json.dumps(["a different question bank"]))
    with pytest.raises(RuntimeError, match="questions mismatch"):
        ana.main(
            [
                "judge",
                "--out-dir",
                str(tmp_path / "o"),
                "--behavior",
                BEH,
                "--panel-root",
                str(pipeline["gen"]["panel_root"]),
                "--base-panel-root",
                str(pipeline["gen"]["base_root"]),
                "--questions",
                str(qpath),
                "--smoke",
                "--stub-judge",
            ]
        )


def test_judge_resume_skips_same_regime(pipeline):
    """Re-running judge under the SAME regime key resumes from the per-cell
    checkpoints (identical aggregate, no error)."""
    out_dir = pipeline["out_dir"]
    before = (out_dir / "panel_aggregate_imp.json").read_text()
    rc = ana.main(
        [
            "judge",
            "--out-dir",
            str(out_dir),
            "--behavior",
            BEH,
            "--panel-root",
            str(pipeline["gen"]["panel_root"]),
            "--base-panel-root",
            str(pipeline["gen"]["base_root"]),
            "--questions",
            str(pipeline["qpath"]),
            "--smoke",
            "--stub-judge",
        ]
    )
    assert rc == 0
    after = json.loads((out_dir / "panel_aggregate_imp.json").read_text())
    assert after["arms"] == json.loads(before)["arms"]


def test_figures_render_end_to_end(pipeline):
    import issue1481_figures as figs

    fig_dir = pipeline["root"] / "figures"
    rc = figs.main(
        [
            "--analysis-dir",
            str(pipeline["out_dir"]),
            "--fig-dir",
            str(fig_dir),
        ]
    )
    assert rc == 0
    expected = [
        "hero1_forest_matched_install",
        "hero1_percell_raw",
        "hero2_marker_emission_map",
        "tier1_ladders_imp",
        "panel_heatmap_imp",
        "per_seed_D_scatter",
        "install_bars_imp",
        "heldout_vs_pooled_decomposition",
        "marker_three_space_divergence",
        "margin_vs_rate_validation",
        "marker_dose_curves",
        "marker_dose_curves_perq_raw",
        "marker_install_trajectories",
    ]
    for stem in expected:
        png = fig_dir / f"{stem}.png"
        assert png.exists() and png.stat().st_size > 0, stem
    assert (fig_dir / "marker_three_space_table.json").exists()


def test_errorbar_offsets_clamp_inverted_ci(pipeline, tmp_path):
    """An INVERTED quantile CI routed through the REAL figure function must
    render (matplotlib rejects negative xerr/yerr; gotchas.md errorbar rule)."""
    import issue1481_figures as figs

    content = json.loads((pipeline["out_dir"] / "regime_contrast_content.json").read_text())
    cell = content["behavior_contexts"][BEH_KEY]["pers"]
    d = cell["pooled"]["D"]
    cell["pooled"]["newcombe_95"] = [d + 1e-7, d - 1e-7]  # inverted around D
    cell["heldout"]["pooled"]["newcombe_95"] = [
        cell["heldout"]["pooled"]["D"] + 1e-7,
        cell["heldout"]["pooled"]["D"] - 1e-7,
    ]
    analysis = {"content": content}
    fig_dir = tmp_path / "figs"
    fig_dir.mkdir()
    figs.fig_hero1_forest(analysis, fig_dir)
    figs.fig_heldout_decomp(analysis, fig_dir)
    assert (fig_dir / "hero1_forest_matched_install.png").exists()
    assert (fig_dir / "heldout_vs_pooled_decomposition.png").exists()


# ── r2 revision coverage: reused-1434 realized shape + Phase B/C wiring ──────


def test_arm_record_reused_1434_fixture_realized_shape(tmp_path):
    """The reused-1434 reader branch against fixtures in the REALIZED
    committed shape — top-level ``"ladders"`` (NOT ``"runs"``) + a
    ``verdict_arms`` sibling — the r1 Critical (#1073
    reused_artifact_schema_drift: the old fixture mirrored the reader's
    assumed shape, so the branch was never exercised)."""
    repo_root = tmp_path / "repo"
    for regime, rel in cells.REUSED_1434_LADDERS.items():
        recs = {}
        for ctx in cells.CTX_KEYS:
            for lr in fu4.FU4_LRS:
                recs[cells.reused_1434_run_id(ctx, regime, lr)] = _run_rec({10: 0.70, 20: 0.80}, 10)
        path = repo_root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"ladders": recs, "verdict_arms": {}}))
    paths = ana.SelectPaths(
        ladders_dir=tmp_path / "no-fresh-ladders", repo_root=repo_root, marker_root=None
    )
    for ctx in cells.CTX_KEYS:
        for regime in cells.REGIMES:
            arm_id, sel, rates, source = ana._arm_record(paths, "cas", ctx, regime, 1e-5, 42)
            assert arm_id == f"cas-{ctx}-{regime}-lr1e5-s42"
            assert source == "reused-1434"
            assert sel["step"] == 10 and rates


def test_reused_1434_arms_resolve_on_real_committed_artifacts():
    """All 24 reused #1434 casual seed-42 arms resolve via ``_arm_record``
    against the REAL committed ladders JSONs (r1 Critical regression pin:
    fails pre-fix with "no 'runs' dict" on the committed ``"ladders"`` key;
    requires the eval_results/issue_1434 cone — tests/sparse_cones.txt)."""
    for rel in cells.REUSED_1434_LADDERS.values():
        assert (REPO / rel).exists(), f"missing committed artifact {rel}"
    paths = ana.SelectPaths(ladders_dir=REPO / "no-such-dir", repo_root=REPO, marker_root=None)
    n = 0
    for ctx in cells.CTX_KEYS:
        for regime in cells.REGIMES:
            for lr in fu4.FU4_LRS:
                arm_id, sel, rates, source = ana._arm_record(paths, "cas", ctx, regime, lr, 42)
                assert source == "reused-1434", arm_id
                assert "step" in sel and rates, arm_id
                n += 1
    assert n == 24


def test_merge_cohort_manifests_union_and_fail_loud(tmp_path):
    """Phase-B prep (r1 Major phase-bc-fresh-arm-wiring): the per-seed cohort
    manifests union into ONE round-wide manifest; a missing cohort or a
    conflicting duplicate run entry fails loud."""
    rname = cells.round_name(BEH_KEY, "con")

    def _write(seed: int, runs: list[dict]) -> None:
        p = tmp_path / f"cell_manifest_{rname}_s{seed}.json"
        p.write_text(json.dumps({"issue": 1481, "round": rname, "runs": runs}))

    _write(42, [{"run_id": "imp-pers-con-lr1e5-s42", "train_mix_sha256": "aa"}])
    _write(137, [{"run_id": "imp-pers-con-lr1e5-s137", "train_mix_sha256": "bb"}])
    merged = wk._merge_cohort_manifests(rname, tmp_path, (42, 137))
    assert {r["run_id"] for r in merged["runs"]} == {
        "imp-pers-con-lr1e5-s42",
        "imp-pers-con-lr1e5-s137",
    }
    assert merged["merged_seeds"] == [42, 137]
    assert merged["merged_from"] == [
        f"cell_manifest_{rname}_s42.json",
        f"cell_manifest_{rname}_s137.json",
    ]
    with pytest.raises(FileNotFoundError):
        wk._merge_cohort_manifests(rname, tmp_path, (42, 999))
    _write(137, [{"run_id": "imp-pers-con-lr1e5-s42", "train_mix_sha256": "CONFLICT"}])
    with pytest.raises(RuntimeError):
        wk._merge_cohort_manifests(rname, tmp_path, (42, 137))


def test_fresh_arm_ckpt_stages_from_hub(tmp_path, monkeypatch):
    """``phase_panel --arms`` on a FRESH Phase-C instance (r1 Major
    phase-bc-fresh-arm-wiring): the build record stages from the run's data
    prefix and the SELECTED rung from the model-repo rung uploads when the
    Phase-A-local ``selected_ckpt`` path does not exist. Real bodies of
    ``_fresh_arm_ckpt`` + ``_stage_fresh_rung`` execute; fakes ONLY at the
    Hub network boundary, signature-conformant by construction."""
    import huggingface_hub as hfh

    cells.register_i1481_rounds()
    rname = cells.round_name(BEH_KEY, "con")
    run = fu4.ROUNDS[rname].runs[0]
    arm_id = run.run_id
    spec = fu4.ROUNDS[rname]
    out_root = tmp_path / "phase_c"
    cfg = wk.run1090.RunConfig(smoke=False, cells=(), out_root=out_root)

    def fake_stage_prefix(prefix, dest, *, skip_if=None):
        assert prefix == f"{spec.data_prefix}/{arm_id}"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{rname}_build_result.json").write_text(
            json.dumps(
                {
                    "status": "trained",
                    "selected_ckpt": "/phase-a-instance/gone/checkpoint-20",
                    "selection": {"step": 20, "rate": 0.70, "in_band": True},
                }
            )
        )

    monkeypatch.setattr(wk.run1090, "_stage_hf_prefix", fake_stage_prefix)
    rung_pir = f"{spec.adapter_prefix}/{arm_id}/checkpoint-20"

    class _FakeApi:
        def list_repo_tree(self, repo_id, path_in_repo=None, repo_type=None, recursive=False):
            assert repo_type == "model" and path_in_repo == rung_pir
            return [
                SimpleNamespace(path=f"{rung_pir}/adapter_config.json"),
                SimpleNamespace(path=f"{rung_pir}/adapter_model.safetensors"),
            ]

    def fake_download(repo_id, filename, *, repo_type=None, local_dir=None):
        p = Path(local_dir) / Path(filename).name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
        return str(p)

    monkeypatch.setattr(hfh, "HfApi", _FakeApi)
    monkeypatch.setattr(hfh, "hf_hub_download", fake_download)

    ckpt = wk._fresh_arm_ckpt(cfg, run, arm_id)
    assert Path(ckpt) == out_root / arm_id / "checkpoint-20"
    assert (Path(ckpt) / "adapter_config.json").exists()
    assert (Path(ckpt) / "adapter_model.safetensors").exists()
    assert not list(Path(ckpt).glob("_hfstage-*")), "staging dir must be reaped"

    # Local fast path: an existing selected_ckpt dir (the Phase-A instance
    # itself) is used verbatim — no Hub staging.
    local_ckpt = tmp_path / "local" / "checkpoint-30"
    local_ckpt.mkdir(parents=True)
    (local_ckpt / "adapter_config.json").write_text("{}")
    run_root = out_root / arm_id
    (run_root / f"{rname}_build_result.json").write_text(
        json.dumps(
            {
                "status": "trained",
                "selected_ckpt": str(local_ckpt),
                "selection": {"step": 30, "rate": 0.70, "in_band": True},
            }
        )
    )
    assert Path(wk._fresh_arm_ckpt(cfg, run, arm_id)) == local_ckpt
