"""Round-22 spread-instrument pins for issue #1739 (evil-ood-spread).

Pins the corrected two-sided spread gate of the compliance full wave:

- BINDING read = per-CONTEXT means, SAMPLE SD (ddof=1), STRICT bottom bin
  (< 10) — the canonical ``gates.score_spread`` / ``gates.per_context_means``
  (instrument-matched to the committed trait-DV verdicts).
- Legacy per-item read retained as a labelled SECONDARY block
  (``spread_per_item_legacy``: per-rollout unit, population SD ddof=0,
  inclusive bottom bin) so previously-published numbers stay traceable.
- ``--reduce-from-raw`` fail-loud raises (never warn-and-continue at rc=0).
- Cross-copy agreement pins (consolidation tripwires) against
  ``gate2_spread_floor`` and ``issue1739_k1_floor.rung_table``.

All fixtures are SYNTHETIC numeric scores invented here — never rows lifted
from the retained judge artifacts (trigger-dense content discipline).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from explore_persona_space.experiments.issue_1739 import gates  # noqa: E402


def _cf():
    import issue1739_compliance_full as cf

    return cf


# ---------------------------------------------------------------------------
# Canonical helper: estimator + unit pins
# ---------------------------------------------------------------------------


def test_sd_is_sample_ddof1_not_population():
    """sd must be SAMPLE SD (ddof=1) — fails under the pre-fix pstdev choice."""
    rep = gates.score_spread([0.0, 10.0, 20.0], unit="per_context")
    assert rep["sd"] == pytest.approx(10.0, abs=1e-12)  # pstdev would be ~8.165
    assert rep["sd_pop"] == pytest.approx(8.16496580927726, abs=1e-9)
    assert rep["sd_ddof"] == 1
    assert rep["sd"] != pytest.approx(rep["sd_pop"], abs=1e-3)


def test_bottom_bin_is_strict_lt():
    """A score EXACTLY at the edge (10.0) is NOT bottom-bin under strict `<`.

    Fails under the pre-fix inclusive ``<=`` choice (which reads 0.5 here).
    """
    rep = gates.score_spread([10.0, 50.0], unit="per_context")
    assert rep["bottom_frac"] == 0.0
    assert rep["bottom_frac_inclusive"] == 0.5
    assert rep["bottom_bin"] == "strict < 10"


def test_score_spread_zero_case_shape():
    rep = gates.score_spread([], unit="per_context")
    assert rep["n_scores"] == 0
    assert rep["sd"] is None and rep["spread_gate_pass"] is False
    assert rep["spread_unit"] == "per_context"


def test_per_context_means_two_level_mean():
    """Per item mean over kept draws, then per context mean over kept items."""
    pcm = gates.per_context_means(
        {
            "ctxA_k00": [0.0, 0.0],
            "ctxA_k01": [100.0, 100.0],
            "ctxB_k00": [50.0, 50.0],
            "ctxB_k01": [None, None],  # all-dropped item: excluded, never coerced
        }
    )
    assert pcm == {"ctxA": 50.0, "ctxB": 50.0}


def test_per_context_means_malformed_item_id_raises():
    with pytest.raises(ValueError):
        gates.per_context_means({"no-k-suffix": [50.0]})


def test_per_context_and_per_item_reads_diverge():
    """Fixture where the two units give DIFFERENT sd — both values pinned."""
    per_item = {
        "ctxA_k00": [0.0, 0.0],  # item mean 0
        "ctxA_k01": [100.0, 100.0],  # item mean 100
        "ctxB_k00": [50.0, 50.0],  # item mean 50
    }
    ctx_vals = list(gates.per_context_means(per_item).values())
    assert ctx_vals == [50.0, 50.0]
    rep_ctx = gates.score_spread(ctx_vals, unit="per_context")
    assert rep_ctx["sd"] == pytest.approx(0.0, abs=1e-12)
    rep_item = gates.score_spread([0.0, 100.0, 50.0], unit="per_item")
    assert rep_item["sd"] == pytest.approx(50.0, abs=1e-9)  # ddof=1 over item means
    assert rep_item["sd_pop"] == pytest.approx(40.824829046386306, abs=1e-9)


# ---------------------------------------------------------------------------
# compliance_full per-rung report: binding + legacy blocks
# ---------------------------------------------------------------------------


def _judged_fixture() -> dict:
    per_item = {
        "ctxA_k00": [0.0, 0.0],
        "ctxA_k01": [100.0, 100.0],
        "ctxB_k00": [50.0, 50.0],
    }
    return {
        "n_items": 3,
        "per_item_scores": per_item,
        "n_dropped_draws_content": 0,
        "n_transport_lost_draws": 0,
        "n_total_draws": 6,
        "judge_raw_path": "unused.json",
    }


def test_reduce_rung_report_binding_vs_legacy(tmp_path):
    cf = _cf()
    rep = cf._reduce_rung_report(_judged_fixture(), trait_root=tmp_path, rung="r")
    # Binding top-level read: per-context, ddof=1, strict bottom bin.
    assert rep["spread_unit"] == "per_context"
    assert rep["sd_ddof"] == 1
    assert rep["n_contexts"] == 2 and rep["n_scored"] == 2
    assert rep["sd"] == pytest.approx(0.0, abs=1e-12)  # ctx means 50, 50
    assert rep["bottom_bin"] == "strict < 10"
    # Legacy SECONDARY block: per-item, pstdev ddof=0, inclusive bottom bin —
    # the previously-published convention, labelled and non-headline.
    legacy = rep["spread_per_item_legacy"]
    assert legacy["spread_unit"] == "per_item"
    assert legacy["sd_ddof"] == 0
    assert legacy["n_items_scored"] == 3
    assert legacy["sd"] == pytest.approx(40.824829046386306, abs=1e-9)
    assert legacy["sd_sample_ddof1"] == pytest.approx(50.0, abs=1e-9)
    assert legacy["bottom_bin"] == "inclusive <= 10"
    # Divergence is recorded machine-readably, never hidden.
    assert "ddof=1" in rep["sd_convention_note"] and "np.std" in rep["sd_convention_note"]


def test_reduce_rung_report_legacy_reproduces_published_convention(tmp_path):
    """Legacy block == pstdev + inclusive `<=` over per-item means (traceability)."""
    import statistics

    cf = _cf()
    per_item = {
        "ctxA_k00": [10.0],  # exactly-edge item: inclusive counts it, strict does not
        "ctxA_k01": [30.0],
        "ctxB_k00": [80.0],
        "ctxC_k00": [5.0],
    }
    judged = dict(_judged_fixture(), per_item_scores=per_item, n_items=4)
    rep = cf._reduce_rung_report(judged, trait_root=tmp_path, rung="r")
    item_means = [10.0, 30.0, 80.0, 5.0]
    legacy = rep["spread_per_item_legacy"]
    assert legacy["sd"] == pytest.approx(statistics.pstdev(item_means), abs=1e-12)
    assert legacy["bottom_frac"] == pytest.approx(2 / 4)  # inclusive: 10.0 and 5.0
    assert legacy["bottom_frac_strict"] == pytest.approx(1 / 4)  # strict: 5.0 only
    assert rep["bottom_frac"] == pytest.approx(1 / 3)  # ctx means 20, 80, 5 → 5 only


# ---------------------------------------------------------------------------
# --reduce-from-raw: fail-loud raises + carry-over + recount reconciliation
# ---------------------------------------------------------------------------

STRUCK_DETECTION = {
    "invalid": True,
    "reason": "struck-sentinel (synthetic)",
    "struck_by": "test-fixture",
}


def _write_manifest(out_path: Path, rung: str, published_rung: dict | None) -> None:
    manifest = {
        "kind": "epm:compliance-dv-results",
        "spread_gate": {"sd_min": 10.0},
        "per_rung": {} if published_rung is None else {rung: published_rung},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(manifest, indent=1))


def _write_raw(out_path: Path, rung: str, all_scores: dict) -> Path:
    raw_path = out_path.parent / "compliance_full" / rung / "judge_raw_compliance_full.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps({"all_scores": all_scores}))
    return raw_path


def _args(tmp_path: Path, rung: str = "rungx") -> argparse.Namespace:
    return argparse.Namespace(
        output=str(tmp_path / "compliance_dv_results.json"),
        trait_root=str(tmp_path / "traits"),  # absent → rho None (unit unchanged)
        rungs=[rung],
        add_rung=[],
    )


def _published_rung_fixture(**overrides) -> dict:
    base = {
        "n_items": 4,
        "n_scored": 4,
        "sd": 1234.5,  # legacy per-item value; echoed, never compared numerically
        "bottom_frac": 0.25,
        "n_total_draws": 8,
        "n_dropped_draws_content": 0,
        "n_transport_lost_draws": 0,
        "detection": dict(STRUCK_DETECTION),
        "rho_compliance_vs_trait": {"rho": None, "p": None, "n_common": 0},
    }
    base.update(overrides)
    return base


def _happy_raw_scores() -> dict:
    # 3 contexts / 4 items / 8 valid draws (synthetic integer scores).
    return {
        "ctxA_k00__00001__01": {"score": 0},
        "ctxA_k00__00001__02": {"score": 0},
        "ctxA_k01__00002__01": {"score": 100},
        "ctxA_k01__00002__02": {"score": 100},
        "ctxB_k00__00003__01": {"score": 50},
        "ctxB_k00__00003__02": {"score": 50},
        "ctxC_k00__00004__01": {"score": 10},
        "ctxC_k00__00004__02": {"score": 10},
    }


def test_reduce_from_raw_missing_manifest_raises(tmp_path):
    cf = _cf()
    with pytest.raises(FileNotFoundError):
        cf._reduce_from_raw(_args(tmp_path))


def test_reduce_from_raw_missing_raw_file_raises(tmp_path):
    cf = _cf()
    args = _args(tmp_path)
    _write_manifest(Path(args.output), "rungx", _published_rung_fixture())
    with pytest.raises(FileNotFoundError):
        cf._reduce_from_raw(args)


def test_reduce_from_raw_zero_rows_raises(tmp_path):
    cf = _cf()
    args = _args(tmp_path)
    _write_manifest(Path(args.output), "rungx", _published_rung_fixture())
    _write_raw(Path(args.output), "rungx", {})
    with pytest.raises(RuntimeError, match="zero raw judge rows"):
        cf._reduce_from_raw(args)


def test_reduce_from_raw_zero_kept_scores_raises(tmp_path):
    cf = _cf()
    args = _args(tmp_path)
    _write_manifest(Path(args.output), "rungx", _published_rung_fixture())
    # Every draw content-drops (score None → parse_fail); kept pool empty.
    _write_raw(
        Path(args.output),
        "rungx",
        {
            "ctxA_k00__00001__01": {"score": None},
            "ctxB_k00__00002__01": {"score": None},
        },
    )
    with pytest.raises(RuntimeError, match="zero kept scores"):
        cf._reduce_from_raw(args)


def test_reduce_from_raw_degenerate_single_group_raises(tmp_path):
    cf = _cf()
    args = _args(tmp_path)
    _write_manifest(Path(args.output), "rungx", _published_rung_fixture())
    # Two items, valid scores, ONE context — the shape that would silently
    # turn per-context into per-everything.
    _write_raw(
        Path(args.output),
        "rungx",
        {
            "ctxA_k00__00001__01": {"score": 40},
            "ctxA_k01__00002__01": {"score": 60},
        },
    )
    with pytest.raises(RuntimeError, match="SINGLE group"):
        cf._reduce_from_raw(args)


def test_reduce_from_raw_malformed_item_id_raises(tmp_path):
    cf = _cf()
    args = _args(tmp_path)
    _write_manifest(Path(args.output), "rungx", _published_rung_fixture())
    _write_raw(Path(args.output), "rungx", {"nok-suffix__00001__01": {"score": 40}})
    with pytest.raises(ValueError, match="malformed rollout item id"):
        cf._reduce_from_raw(args)


def test_reduce_from_raw_happy_path_carries_struck_detection_verbatim(tmp_path):
    cf = _cf()
    args = _args(tmp_path)
    out_path = Path(args.output)
    _write_manifest(out_path, "rungx", _published_rung_fixture())
    _write_raw(out_path, "rungx", _happy_raw_scores())

    assert cf._reduce_from_raw(args) == 0
    manifest = json.loads(out_path.read_text())
    rep = manifest["per_rung"]["rungx"]
    # Struck detection block carried over VERBATIM (never recomputed).
    assert rep["detection"] == STRUCK_DETECTION
    # Binding read: ctx means [50, 50, 10] → ddof=1 sd; strict bottom excludes 10.0.
    assert rep["spread_unit"] == "per_context" and rep["sd_ddof"] == 1
    assert rep["n_contexts"] == 3
    assert rep["sd"] == pytest.approx(23.094010767585033, abs=1e-9)
    assert rep["bottom_frac"] == 0.0  # 10.0 is NOT strictly < 10
    # Legacy block traceable: per-item means [0, 100, 50, 10], pstdev.
    legacy = rep["spread_per_item_legacy"]
    assert legacy["n_items_scored"] == 4 and legacy["sd_ddof"] == 0
    assert legacy["bottom_frac"] == pytest.approx(2 / 4)  # inclusive: 0 and 10
    # Drop recount matches the synthetic published counts.
    assert rep["published_recount"]["match"] is True
    assert rep["published_recount"]["mismatches"] == {}
    # Realized selection recorded (the reduction demonstrably reduced).
    sel = manifest["reduce_from_raw"]["per_rung_selection"]["rungx"]
    assert sel == {
        "n_raw_rows": 8,
        "n_items": 4,
        "n_items_scored": 4,
        "n_contexts": 3,
        "n_kept_draws": 8,
    }
    # Manifest-level convention fields are machine-readable.
    sg = manifest["spread_gate"]
    assert sg["spread_unit"] == "per_context" and sg["sd_ddof"] == 1
    assert sg["bottom_bin"] == "strict < 10"


def test_reduce_from_raw_recount_mismatch_reported_not_overwritten(tmp_path):
    cf = _cf()
    args = _args(tmp_path)
    out_path = Path(args.output)
    _write_manifest(
        out_path, "rungx", _published_rung_fixture(n_total_draws=9999)
    )  # deliberate mismatch
    _write_raw(out_path, "rungx", _happy_raw_scores())

    assert cf._reduce_from_raw(args) == 0
    manifest = json.loads(out_path.read_text())
    rep = manifest["per_rung"]["rungx"]
    recount = rep["published_recount"]
    assert recount["match"] is False
    assert recount["mismatches"]["n_total_draws"] == {"published": 9999, "recount": 8}
    # The recount stays in place AND the published value stays visible.
    assert rep["n_total_draws"] == 8


# ---------------------------------------------------------------------------
# Consolidation agreement pins (drift tripwires across the remaining copies)
# ---------------------------------------------------------------------------


def test_compliance_full_wrapper_agrees_with_canonical(tmp_path):
    cf = _cf()
    vals = [3.0, 9.99, 10.0, 40.0, 95.0]
    rep_script = cf._canonical_spread(vals, unit="per_context")
    rep_canon = gates.score_spread(vals, unit="per_context")
    assert rep_script == rep_canon


def test_pilot_judge_wrapper_agrees_with_canonical():
    import issue1739_pilot_judge as pj

    vals = [3.0, 9.99, 10.0, 40.0, 95.0]
    rep_pj = pj._score_spread(vals, unit="context")
    rep_canon = gates.score_spread(vals, unit="context")
    assert rep_pj == rep_canon


def test_gate2_spread_floor_agrees_with_canonical():
    """gate2_spread_floor keeps its own arithmetic (committed-verdict producer);
    this pin trips if the two ever diverge again."""
    vals = [3.0, 9.99, 10.0, 40.0, 95.0]
    rows = [{"context_id": f"c{i}", "dv": v} for i, v in enumerate(vals)]
    g2 = gates.gate2_spread_floor(rows, behavior="synthetic")
    canon = gates.score_spread(vals, unit="per_context")
    assert g2["inter_context_sd"] == pytest.approx(canon["sd"], abs=1e-9)
    assert g2["bottom_bin_frac"] == pytest.approx(canon["bottom_frac"], abs=1e-12)


def test_k1_floor_rung_table_agrees_with_canonical():
    """k1_floor.rung_table keeps its own arithmetic (committed k1 verdicts);
    this pin trips if the two ever diverge again (rounded fields)."""
    import issue1739_k1_floor as k1

    vals = [3.0, 9.99, 10.0, 40.0, 95.0]
    rows = [{"rung": "r", "dv": v} for v in vals]
    table = k1.rung_table(rows, sd_floor=10.0, bottom_edge=10.0, bottom_max=0.80)
    canon = gates.score_spread(vals, unit="per_context")
    assert table["r"]["dv_sd"] == pytest.approx(round(canon["sd"], 3), abs=1e-9)
    assert table["r"]["bottom_bin_frac"] == pytest.approx(round(canon["bottom_frac"], 4), abs=1e-9)
