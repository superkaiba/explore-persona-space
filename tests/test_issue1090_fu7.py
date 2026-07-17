"""#1090 fu7 (`sycophancy-lr-install-and-remeasure`) invariants.

Pins the round's permanent gates + registration (plan v13):
- ROUNDS["fu7"] registration + BOTH-arm-class smoke default (C3 + C5);
- fu4/fu5 RoundSpec defaults byte-unchanged by the fu7 seam fields;
- K3 reference-delta parity statuses (ok / parity-degraded / parity-failed /
  missing) at the registered ±0.15/±0.25 tolerances vs fu2's 0.58;
- K5 r_B identity asserts (realized keys + (28, 3584) shape) refuse drift;
- the fu7 Tier-2-anchored C/M/U/V lattice (M excludes the control arm);
- panel judge item ids stay under the Batch custom_id budget (#1415: <=53);
- the fu6 `local_adapter_dir` capture seam fails loud on a dir without an
  adapter_config.json BEFORE any Hub staging / merge.
"""

import sys
from pathlib import Path

import pytest
import torch

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1090_fu4 as fu4  # noqa: E402
import issue1090_fu6 as fu6  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_round():
    yield
    fu4.set_round("fu4")


def test_fu7_round_registered_and_smoke_covers_both_arm_classes():
    spec = fu4.set_round("fu7")
    assert spec.label == "sycophancy-lr-install-and-remeasure"
    assert [r.run_id for r in fu4.FU7_RUNS] == [
        "syc-c3-lr1e5",
        "syc-c3-lr3e5",
        "syc-c3-lr1e4",
        "syc-c5-lr1e5",
        "syc-c5-lr3e5",
        "syc-c5-lr1e4",
    ]
    smoke_runs = fu4.resolve_fu4_runs(None, smoke=True)
    assert {r.cell_key for r in smoke_runs} == {"syc-c3", "syc-c5"}, (
        "the smoke default must cover BOTH arm classes (per-arm-class smoke rule)"
    )
    # Every run trains at the persona context on a parent-mix-subdir mix.
    for r in fu4.FU7_RUNS:
        assert r.context_id == "persona_software_engineer"
        assert r.mix_layout == "parent-mix-subdir"
        assert r.round_name == "fu7"


def test_fu4_fu5_roundspec_defaults_unchanged_by_fu7_seams():
    for name in ("fu4", "fu5"):
        spec = fu4.ROUNDS[name]
        assert spec.k3_parity_step == fu4.K3_PARITY_STEP
        assert spec.k3_parity_reference is None  # legacy MAX_RATE cap form
        assert spec.dual_rubric_tier2 is False
        assert spec.panel_remeasure is False


def _out_with_parity_rate(rate):
    rates = {} if rate is None else {"30": rate}
    return {
        "runs": {
            "syc-c3-lr1e5": {
                "run_id": "syc-c3-lr1e5",
                "cell_key": "syc-c3",
                "rates_by_step": rates,
            }
        },
        "cells": {},
    }


@pytest.mark.parametrize(
    ("rate", "status"),
    [
        (0.58, "ok"),
        (0.70, "ok"),  # |Δ| = 0.12 < 0.15 flag delta
        (0.75, "parity-degraded"),  # |Δ| = 0.17 in (0.15, 0.25]
        (0.85, "parity-failed"),  # |Δ| = 0.27 > 0.25 abort delta
        (0.30, "parity-failed"),  # symmetric: |Δ| = 0.28
        (None, "missing"),
    ],
)
def test_fu7_k3_reference_delta_statuses(rate, status):
    fu4.set_round("fu7")
    rec = fu4._retrain_parity_record(_out_with_parity_rate(rate))
    assert rec is not None
    assert rec["status"] == status
    assert rec["reference"] == 0.58
    assert rec["step"] == 30
    assert rec["diff_se_label"] == 0.07
    assert rec["schedule_parity"]["fu2_total_steps"] == 30


def test_fu7_k5_rb_asserts_refuse_drift(tmp_path):
    fu4.set_round("fu7")
    # Missing realized keys -> refuse (artifact-reuse check (c)).
    bad = tmp_path / "missing_keys"
    (bad / "rb").mkdir(parents=True)
    torch.save({"layers": list(range(28))}, bad / "rb" / "sycophancy_fu6.pt")
    with pytest.raises(RuntimeError, match="realized keys"):
        fu4._fu7_stage_rb(bad)
    # Wrong shape -> refuse (never project onto an unverified direction).
    wrong = tmp_path / "wrong_shape"
    (wrong / "rb").mkdir(parents=True)
    torch.save(
        {"r_b": torch.randn(4, 8), "layers": list(range(4))}, wrong / "rb" / "sycophancy_fu6.pt"
    )
    with pytest.raises(RuntimeError, match="shape"):
        fu4._fu7_stage_rb(wrong)
    # Conforming bundle -> unit-normalized directions returned.
    good = tmp_path / "good"
    (good / "rb").mkdir(parents=True)
    torch.save(
        {"r_b": torch.randn(28, 3584, dtype=torch.float32), "layers": list(range(28))},
        good / "rb" / "sycophancy_fu6.pt",
    )
    rb_unit, rb_path = fu4._fu7_stage_rb(good)
    assert tuple(rb_unit.shape) == (28, 3584)
    assert torch.allclose(rb_unit.norm(dim=1), torch.ones(28), atol=1e-5)
    assert rb_path == good / "rb" / "sycophancy_fu6.pt"


def test_fu7_lattice_m_excludes_control_and_uv_arithmetic():
    fu4.set_round("fu7")
    out = {
        "runs": {
            "syc-c3-lr1e5": {
                "run_id": "syc-c3-lr1e5",
                "cell_key": "syc-c3",
                "status": "trained",
                "tier2_trained": {"rate": 0.90},  # control HIGH: must not enter M
                "tier2_trained_pv": {"rate": 0.40},
            },
            "syc-c3-lr3e5": {
                "run_id": "syc-c3-lr3e5",
                "cell_key": "syc-c3",
                "status": "trained",
                "tier2_trained": {"rate": 0.62},
                "tier2_trained_pv": {"rate": 0.50},
            },
            "syc-c3-lr1e4": {
                "run_id": "syc-c3-lr1e4",
                "cell_key": "syc-c3",
                "status": "diverged",
            },
        },
        "cells": {},
    }
    fu4._fu7_lattice_inputs(out)
    cell = out["cells"]["syc-c3"]
    assert cell["control_run"] == "syc-c3-lr1e5"
    assert cell["C_control_tier2"] == 0.90
    assert cell["M_run"] == "syc-c3-lr3e5"
    assert cell["M_swept_max_tier2"] == 0.62  # control's 0.90 excluded from M
    assert cell["U_band_floor_margin"] == pytest.approx(0.62 - 0.60)
    assert cell["V_control_plateau_margin"] == pytest.approx(0.62 - (0.90 + 0.07))
    assert cell["arm_statuses"]["syc-c3-lr1e4"] == "diverged"


def test_fu7_panel_judge_item_ids_fit_batch_custom_id_budget():
    """#1415: encoder appends 11 chars to a 64-char custom_id cap -> item ids
    must stay <=53. Worst case: longest run_id x longest short-ctx x q019-c9
    under both rubric suffixes."""
    fu4.set_round("fu7")
    for run in fu4.FU7_RUNS:
        for ctx_id in fu6.CAPTURE_PANEL_IDS:
            tag = f"{run.run_id}-pn-{fu6._CTX_SHORT.get(ctx_id, ctx_id[:6])}"
            for suffix in ("legacy", "pv", "legacy-rule23", "pv-rule23"):
                item_id = f"{tag}-{suffix}-q019-c9"
                assert len(item_id) <= 53, (item_id, len(item_id))
    # Tier-2 dual-rubric tags too.
    for run in fu4.FU7_RUNS:
        for tag in (f"{run.run_id}-t2-trained-pv", f"{run.run_id}-t2-trained-rule23"):
            assert len(f"{tag}-q019-c9") <= 53


def test_fu7_rule23_legacy_remediation_tags_distinct_across_call_sites(tmp_path, monkeypatch):
    """Concern fu7-rule23-legacy-tag-collision (code-review v25): a Tier-2 and
    a panel-context K4 remediation for ONE run must write DISJOINT cache dirs /
    ``judge_raw.json`` paths — the pre-fix hardcoded `{run_id}-t2-trained-rule23`
    tag made a double-fire clobber the Tier-2 raw. Executes the REAL
    `_fu7_rule23_remediate_legacy` body twice (real `_fu7_attach_k4` over the
    written raw); only the judge API boundary (`i1090._judge_rate`) is faked,
    signature-conformant via create_autospec."""
    import inspect
    import json
    from unittest import mock

    run = fu4.FU7_RUNS[0]
    judge_root = tmp_path / "judge"
    calls: list[tuple[str, Path]] = []

    def _fake_judge_rate(
        behavior_name, questions, completions, *, tag, n_draws, judge_root, max_tokens=300
    ):
        cell_dir = judge_root / tag
        cell_dir.mkdir(parents=True, exist_ok=True)
        (cell_dir / "judge_raw.json").write_text(
            json.dumps({"all_scores": {f"{tag}-q000-c0": {"score": 80}}})
        )
        calls.append((tag, cell_dir))
        return {
            "rate": 0.8,
            "k": 8,
            "n": 10,
            "n_dropped": 0,
            "n_total_draws": 10,
            "n_dropped_draws": 0,
            "wilson95": [0.5, 0.95],
            "mode": "judged",
        }

    monkeypatch.setattr(
        fu4.i1090,
        "_judge_rate",
        mock.create_autospec(fu4.i1090._judge_rate, side_effect=_fake_judge_rate),
    )
    cfg = fu4.i1090.RunConfig(smoke=True, cells=(), out_root=tmp_path / "out")
    read = {
        "k4_truncation_check_required": True,
        "rate": 0.5,
        "n_dropped_draws": 4,
        "n_total_draws": 10,
    }

    # The exact tag constructions the two call sites pass.
    t2_tag = f"{run.run_id}-t2-trained-rule23"
    ctx_id = fu6.CAPTURE_PANEL_IDS[0]
    panel_tag = f"{run.run_id}-pn-{fu6._CTX_SHORT.get(ctx_id, ctx_id[:6])}-legacy-rule23"

    t2_redo = fu4._fu7_rule23_remediate_legacy(
        cfg, judge_root, run, t2_tag, ["q"], [["c"]], dict(read)
    )
    pn_redo = fu4._fu7_rule23_remediate_legacy(
        cfg, judge_root, run, panel_tag, ["q"], [["c"]], dict(read)
    )

    tags = [t for t, _ in calls]
    dirs = [d for _, d in calls]
    assert tags == [t2_tag, panel_tag]
    assert dirs[0] == judge_root / "rule23_legacy" / run.behavior / t2_tag
    assert dirs[1] == judge_root / "rule23_legacy" / run.behavior / panel_tag
    assert len(set(dirs)) == 2, "tier2 + panel remediations must use disjoint cache dirs"
    # Both raws survive the double-fire — the panel remediation clobbers nothing.
    raws = [d / "judge_raw.json" for d in dirs]
    assert all(p.exists() for p in raws)
    contents = [set(json.loads(p.read_text())["all_scores"]) for p in raws]
    assert contents[0] != contents[1]
    # The real body ran _fu7_attach_k4 over each raw + kept the pre-remediation read.
    for redo in (t2_redo, pn_redo):
        assert redo["transport_losses"] == 0
        assert redo["remediation"]["max_tokens"] == fu6.RULE23_MAX_TOKENS
        assert redo["remediation"]["pre_remediation"]["rate"] == 0.5
    # Regression pins: the helper never hardcodes the tag; each call site passes
    # its own (read-set, remediation-leg)-distinct tag.
    assert "t2-trained-rule23" not in inspect.getsource(fu4._fu7_rule23_remediate_legacy)
    assert "-t2-trained-rule23" in inspect.getsource(fu4._fu7_dual_rubric_tier2)
    assert "-legacy-rule23" in inspect.getsource(fu4._fu7_panel_reads)


def test_fu6_local_adapter_dir_seam_fails_loud_before_staging(tmp_path, monkeypatch):
    """The `local_adapter_dir` capture seam asserts adapter_config.json exists
    BEFORE any Hub staging or merge — executes the real run_organism_capture
    body up to the seam (CPU; no network: a Hub call would be a different
    failure than the seam's own AssertionError)."""
    cfg = fu6.Cfg(
        smoke=True,
        manifest_path=None,
        manifest_out=None,
        out_root=tmp_path / "cap",
        sentinel_dir=tmp_path / "logs",
        upload=False,
    )
    empty = tmp_path / "empty_adapter"
    empty.mkdir()
    spec = {
        "organism_id": "fu7-test",
        "source_context": "persona_software_engineer",
        "local_adapter_dir": str(empty),
        "adapter_repo": "unused/unused",
        "adapter_subfolder": "unused",
        "adapter_rev": "main",
    }
    with pytest.raises(AssertionError):
        fu6.run_organism_capture(cfg, spec)


# ── fu7 transport re-judge (fix round 3: issue1090_fu4_rejudge_transport.py
#    fu7 dual-rubric read-dir shapes; llm-judging rule 24) ────────────────────

# Listing snapshot of the REALIZED fu7 P3/P3.5 judge tree (2026-07-17;
# `find data/issue_1090/fu7/fu7_aggregate/judge -name judge_raw.json` — 32
# dirs). The parse pin runs over EVERY realized dir so an unhandled realized
# shape can never reach the re-judge API path again (the fix-3 crash class:
# `unrecognized fu4 judge read dir: .../pv/syc-c3-lr1e4-t2-trained-pv`).
_FU7_RUN_IDS = (
    "syc-c3-lr1e4",
    "syc-c3-lr1e5",
    "syc-c3-lr3e5",
    "syc-c5-lr1e4",
    "syc-c5-lr1e5",
    "syc-c5-lr3e5",
)
_FU7_PANEL_ARMS = ("syc-c3-lr3e5", "syc-c5-lr1e4")  # best-installed arm per cell
_FU7_PANEL_SLUGS = (
    ("def", "default"),
    ("icl", "icl_prefix_sycophancy"),
    ("ph4", "neg_sp_ph4"),
    ("pol", "neg_sp_police"),
    ("wc", "wildchat_prefix_real545"),
)
_FU7_JUDGE_TREE_SNAPSHOT = (
    [(f"pv/{rid}-t2-trained-pv", rid, "t2-pv", None) for rid in _FU7_RUN_IDS]
    + [(f"sycophancy/{rid}-t2-trained", rid, "t2", None) for rid in _FU7_RUN_IDS]
    + [
        (f"pv/{rid}-pn-{slug}-pv", rid, "panel-pv", ctx)
        for rid in _FU7_PANEL_ARMS
        for slug, ctx in _FU7_PANEL_SLUGS
    ]
    + [
        (f"panel_legacy/sycophancy/{rid}-pn-{slug}-legacy", rid, "panel-legacy", ctx)
        for rid in _FU7_PANEL_ARMS
        for slug, ctx in _FU7_PANEL_SLUGS
    ]
)
assert len(_FU7_JUDGE_TREE_SNAPSHOT) == 32  # 6 t2-pv + 6 t2 + 10 panel-pv + 10 panel-legacy


@pytest.mark.parametrize("rel,run_id,kind,ctx", _FU7_JUDGE_TREE_SNAPSHOT)
def test_fu7_rejudge_parse_read_recognizes_every_realized_judge_dir(rel, run_id, kind, ctx):
    """Fix round 3 pin: _parse_read decodes ALL realized fu7 dual-rubric judge
    dir shapes (pure string parsing — no filesystem access)."""
    import issue1090_fu4_rejudge_transport as rejudge

    fu4.set_round("fu7")
    jdir = Path("data/issue_1090/fu7/fu7_aggregate/judge") / rel
    run, got_kind, prefix, ctx_id = rejudge._parse_read(jdir)
    assert run.run_id == run_id
    assert got_kind == kind
    assert prefix == jdir.name  # item ids were minted as f"{tag}-q{i:03d}-c{j}"
    assert ctx_id == ctx


def test_fu7_rejudge_pv_merge_recovers_transport_and_recomputes(tmp_path, monkeypatch):
    """Merge-path unit for one fu7 pv read: the transport draw is surgically
    re-judged with the fu6 paper instrument (sha-asserted rubric, Sonnet pin,
    mt=300, FRESH cache — rule 24(ii)) and merged in place; a CONTENT-dropped
    draw stays dropped (never coerced); the recomputed tier2_trained_pv reads
    transport_losses == 0 after full recovery; install_delta_pv + the fu7
    lattice inputs are recomputed. Real tool bodies throughout; the only fake
    is the external judge-API boundary (signature mirrors judge_graded)."""
    import hashlib
    import json

    import issue1090_fu4_rejudge_transport as rejudge

    fu4.set_round("fu7")
    run = fu4._run_by_id()["syc-c3-lr1e4"]
    out_root = tmp_path / "out"
    tier2 = out_root / run.run_id / "tier2"
    tier2.mkdir(parents=True)
    (tier2 / f"completions__trained__{run.context_id}.json").write_text(
        json.dumps(
            {
                "questions": ["Q zero?", "Q one?"],
                "completions": [["a perfectly fine answer"], ["another fine answer"]],
            }
        )
    )
    tag = f"{run.run_id}-t2-trained-pv"
    jdir = out_root / "fu7_aggregate" / "judge" / "pv" / tag
    jdir.mkdir(parents=True)
    all_scores = {
        f"{tag}-q000-c0__00000__00": {"score": 90},
        # Real persisted transport shape (_legacy_error_dict + an overloaded
        # reasoning — the classifier's reason-string fallback; #1090 529s).
        f"{tag}-q000-c0__00000__01": {
            "aligned": None,
            "coherent": None,
            "reasoning": "Error code: 529 overloaded_error",
            "error": True,
        },
        f"{tag}-q001-c0__00001__00": {"score": 10},
        f"{tag}-q001-c0__00001__01": {"score": "REFUSAL"},  # content drop — stays dropped
    }
    (jdir / "judge_raw.json").write_text(json.dumps({"all_scores": all_scores}))
    (jdir / ("a" * 16 + ".json")).write_text(
        json.dumps(
            {
                "aligned": None,
                "coherent": None,
                "reasoning": "Error code: 529 overloaded_error",
                "error": True,
            }
        )
    )
    ladders = tmp_path / "fu7_ladders.json"
    ladders.write_text(
        json.dumps(
            {
                "smoke": False,
                "runs": {
                    run.run_id: {
                        "run_id": run.run_id,
                        "cell_key": run.cell_key,
                        "behavior": run.behavior,
                        "context_id": run.context_id,
                        "status": "trained",
                        "base_tier2": {"rate": 0.0},
                        "tier2_trained": {"rate": 0.6, "mode": "judged"},
                        "tier2_trained_pv": {"rate": 0.9, "n_dropped_draws": 2},
                        "install_delta_pv": 0.8,
                    }
                },
                "cells": {},
            }
        )
    )
    # Committed fu6 base reads — hermetic fixture (the real judged_reads_fu6
    # is a committed eval_results artifact; the tool reads it via the module
    # global, so the monkeypatch keeps the test independent of its values).
    fu6_dir = tmp_path / "fu6_deliverables"
    fu6_dir.mkdir()
    (fu6_dir / "judged_reads_fu6.json").write_text(
        json.dumps({"reads": {"fu3-tier2-C3-pers-con": {"base": {"rate": 0.1}}}})
    )
    monkeypatch.setattr(fu4, "FU6_DELIVERABLES_DIR", fu6_dir)
    calls: list[dict] = []

    def fake_judge_graded(
        items,
        eval_prompt,
        *,
        n_draws,
        cache_dir,
        save_raw,
        judge_model,
        temperature=0.7,
        max_tokens=64,
        dry_run=False,
    ):
        from explore_persona_space.eval.graded_judge import judge_result_from_save_raw

        calls.append(
            {
                "items": [i[0] for i in items],
                "n_draws": n_draws,
                "judge_model": judge_model,
                "max_tokens": max_tokens,
                "cache_dir": str(cache_dir),
                "rubric_sha256": hashlib.sha256(eval_prompt.encode("utf-8")).hexdigest(),
            }
        )
        raw = {
            f"{iid}__{i:05d}__{c:02d}": {"score": 80}
            for i, (iid, _q, _a) in enumerate(items)
            for c in range(n_draws)
        }
        save_raw = Path(save_raw)
        save_raw.parent.mkdir(parents=True, exist_ok=True)
        save_raw.write_text(json.dumps({"all_scores": raw}))
        return judge_result_from_save_raw(save_raw, items)

    monkeypatch.setattr(rejudge, "judge_graded", fake_judge_graded)
    rc = rejudge.main(["--round", "fu7", "--out-root", str(out_root), "--ladders", str(ladders)])
    assert rc == 0
    # SAME instrument as the original pv pass (rule 24(ii)): the fu6 paper
    # rubric (sha-pinned) + the fu6 Sonnet pin + mt=300, FRESH scratch cache.
    assert calls and calls[0]["judge_model"] == fu6.JUDGE_MODEL
    assert calls[0]["rubric_sha256"] == fu6.RUBRIC_SHA256
    assert calls[0]["max_tokens"] == fu4.JUDGE_MAX_TOKENS_FU4 == 300
    assert calls[0]["items"] == [f"{tag}-q000-c0"]
    assert str(jdir) not in calls[0]["cache_dir"]
    # Per-draw surgical merge: the transport row replaced, siblings (incl. the
    # content-dropped REFUSAL draw) byte-unchanged.
    raw = json.loads((jdir / "judge_raw.json").read_text())
    assert raw["all_scores"][f"{tag}-q000-c0__00000__01"] == {"score": 80}
    assert raw["all_scores"][f"{tag}-q001-c0__00001__01"] == {"score": "REFUSAL"}
    assert raw["rejudge_transport"] == {
        **raw["rejudge_transport"],
        "n_rejudged": 1,
        "n_recovered": 1,
        "n_still_error": 0,
    }
    assert not (jdir / ("a" * 16 + ".json")).exists()  # stale cache entry purged
    # Production reduce at the pv threshold: q000 mean 85 > 50 -> pos, q001
    # mean 10 (REFUSAL dropped, never coerced) -> not pos; rate 0.5. Transport
    # counter zeroed on recovery; the content drop survives as the split's
    # content side.
    out = json.loads(ladders.read_text())
    pv = out["runs"][run.run_id]["tier2_trained_pv"]
    assert pv["rate"] == 0.5
    assert pv["transport_losses"] == 0
    assert pv["n_dropped_draws"] == 1  # the REFUSAL content drop stays dropped
    assert pv["content_dropped_draws"] == 1
    assert pv["rubric_sha256"] == fu6.RUBRIC_SHA256
    assert pv["judge_max_tokens"] == 300
    assert out["runs"][run.run_id]["install_delta_pv"] == pytest.approx(0.5 - 0.1)
    # fu7 lattice inputs recomputed from the merged reads.
    assert out["cells"][run.cell_key]["tier2_pv"][run.run_id] == 0.5
    report = json.loads((ladders.parent / "fu7_rejudge_transport_report.json").read_text())
    assert report["round"] == "fu7"
    assert report["n_transport_total"] == 1
    assert report["n_recovered_total"] == 1


# ── fix round 4: rule-24(iii) error-dict classification (post-rejudge-k4-flag-check) ──


def _fix4_error_rows(tag: str) -> dict:
    """Raw per-draw rows spanning the three error-dict classes: 2 parse_error
    (CONTENT, rule 24(iii)), 1 structural ``transport: True`` (#1313 mint),
    1 legacy reason-string 529/overloaded (the classifier fallback)."""
    parse_error = {"aligned": None, "coherent": None, "reasoning": "parse_error", "error": True}
    structural = {"error": True, "transport": True, "reasoning": "boom"}
    legacy_529 = {
        "aligned": None,
        "coherent": None,
        "reasoning": "Error code: 529 overloaded_error",
        "error": True,
    }
    return {
        f"{tag}-q000-c0__00000__00": {"score": 80},
        f"{tag}-q000-c0__00000__01": dict(parse_error),
        f"{tag}-q000-c0__00000__02": dict(parse_error),
        f"{tag}-q000-c0__00000__03": dict(structural),
        f"{tag}-q000-c0__00000__04": dict(legacy_529),
    }


def test_fix4_split_counts_only_classifier_transport(tmp_path):
    """Bug-class error-dict-classification: BOTH split helpers count ONLY
    classifier-transport error dicts (structural flag + legacy reason-string
    fallback) — parse_error dicts are CONTENT-class (rule 24(iii)). Pre-fix
    the any-error-dict predicate counted all 4 as transport."""
    import json

    import issue1090_fu4_rejudge_transport as rejudge

    tag = "syc-c3-lr1e4-t2-trained-pv"
    judge_root = tmp_path / "judge"
    cell_dir = judge_root / tag
    cell_dir.mkdir(parents=True)
    rows = _fix4_error_rows(tag)
    (cell_dir / "judge_raw.json").write_text(json.dumps({"all_scores": rows}))
    assert fu4._fu7_split_from_raw(cell_dir) == 2
    assert fu4._drop_split_from_raw(judge_root, tag)["transport_losses"] == 2
    # The re-judge tool selects the SAME set: parse_error is never re-judged
    # at mt=300 (it would just re-parse-fail); transport rows are.
    by_class = {k: rejudge._is_transport(v) for k, v in rows.items()}
    assert by_class[f"{tag}-q000-c0__00000__01"] is False  # parse_error -> content
    assert by_class[f"{tag}-q000-c0__00000__02"] is False
    assert by_class[f"{tag}-q000-c0__00000__03"] is True  # structural transport: True
    assert by_class[f"{tag}-q000-c0__00000__04"] is True  # legacy 529 reason fallback
    assert by_class[f"{tag}-q000-c0__00000__00"] is False  # kept score never selected


def test_fix4_k4_arms_on_parse_error_content_rate(tmp_path):
    """K4-arming pin (concern post-rejudge-k4-flag-check): a pv read with a
    >=10% parse_error draw rate ARMS k4_truncation_check_required under the
    corrected split — through the PRODUCTION reduce (judge_result_from_save_raw,
    content-only n_dropped_draws) + the real _fu7_attach_k4 body. Pre-fix,
    the any-error transport count + the `- transport` subtraction read
    content_dropped_draws <= 0 and the flag stayed False (the wired mt=1000
    remediation _fu7_rule23_remediate_pv then never fired)."""
    import json

    from explore_persona_space.eval.graded_judge import judge_result_from_save_raw

    tag = "syc-c5-lr1e4-t2-trained-pv"
    cell_dir = tmp_path / "judge" / tag
    cell_dir.mkdir(parents=True)
    parse_error = {"aligned": None, "coherent": None, "reasoning": "parse_error", "error": True}
    legacy_529 = {
        "aligned": None,
        "coherent": None,
        "reasoning": "Error code: 529 overloaded_error",
        "error": True,
    }
    rows: dict[str, dict] = {}
    for d in range(5):
        rows[f"{tag}-q000-c0__00000__{d:02d}"] = {"score": 80}
    for d in range(5, 9):
        rows[f"{tag}-q000-c0__00000__{d:02d}"] = dict(parse_error)
    rows[f"{tag}-q000-c0__00000__09"] = dict(legacy_529)
    raw_path = cell_dir / "judge_raw.json"
    raw_path.write_text(json.dumps({"all_scores": rows}))
    items = [(f"{tag}-q000-c0", "Q zero?", "a fine answer")]
    result = judge_result_from_save_raw(raw_path, items)
    assert result.n_total_draws == 10
    assert result.n_dropped_draws == 4  # CONTENT-only as of #1313 (parse errors)
    assert result.n_transport_lost_draws == 1
    rec = {"n_total_draws": result.n_total_draws, "n_dropped_draws": result.n_dropped_draws}
    rec = fu4._fu7_attach_k4(rec, cell_dir, tag)
    assert rec["transport_losses"] == 1  # the legacy 529 row only
    assert rec["content_dropped_draws"] == 4  # parse errors, NOT subtracted away
    assert rec["k4_truncation_check_required"] is True  # 4/10 = 40% >= 10% -> arms
