"""Issue #2479 U4 — instrument-gate arithmetic, masking, drop accounting.

Pins (hermetic — tmp_path fixtures, zero network / API calls):

(a) gate arithmetic incl. BOUNDARY EQUALITY: verbatim-flatness spread vs
    0.5 x realized-axis-range (spread == threshold PASSes), name-mask shift
    (== 8.0 PASSes) + rank-corr (== 0.7 PASSes), and the freeze module's
    band-agreement (rho >= 0.5) + axis-range (>= 8.0) gates;
(b) masking correctness: word-boundary, case-insensitive match with
    sentence-position case preservation, possessives, and NO cross-name
    collisions over the REAL committed panel names;
(c) drop-accounting propagation: leg-report drop counts + per-item
    all-draws-dropped exclusions ride into instrument_gates.json verbatim
    via the production save_raw reduce (judge_result_from_save_raw).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPTS = REPO / "scripts"
PANEL_JSON = REPO / "eval_results" / "issue_2479" / "panel.json"

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1345_onpolicy_judge_legs as jl  # noqa: E402
import issue2479_freeze_axis as fz  # noqa: E402
import issue2479_instrument_gates as ig  # noqa: E402

PANEL = json.loads(PANEL_JSON.read_text())
NAMES = [r["display_name"] for r in PANEL]
INSERTED = [r for r in PANEL if r["variant_inserted"]]
EXTREMES = [r for r in PANEL if r["design_band"] in ("A", "D")]


# ---------------------------------------------------------------------------
# (a) gate arithmetic — boundary equality cases
# ---------------------------------------------------------------------------
def test_flatness_verdict_boundary() -> None:
    # spread == 0.5 * range is a PASS (<=); the smallest excess FAILs.
    assert ig.flatness_verdict(10.0, 20.0) is True
    assert ig.flatness_verdict(10.000001, 20.0) is False
    assert ig.flatness_verdict(0.0, 0.0) is True  # degenerate: equal means, zero range


def test_flatness_gate_block() -> None:
    block = ig.flatness_gate({"a": 79.0, "b": 80.0, "c": 82.0}, realized_axis_range=20.0)
    assert block["spread"] == pytest.approx(3.0)
    assert block["threshold"] == pytest.approx(10.0)
    assert block["verbatim_flatness_pass"] is True
    tight = ig.flatness_gate({"a": 70.0, "b": 82.0}, realized_axis_range=20.0)
    assert tight["spread"] == pytest.approx(12.0)
    assert tight["verbatim_flatness_pass"] is False


def test_name_mask_verdict_boundaries() -> None:
    assert ig.name_mask_verdict(8.0, 0.7) is True  # both boundaries inclusive
    assert ig.name_mask_verdict(8.0000001, 1.0) is False  # shift just over
    assert ig.name_mask_verdict(0.0, 0.6999999) is False  # corr just under
    assert ig.name_mask_verdict(0.0, 0.7) is True


def test_name_mask_gate_uniform_shift_passes_at_boundary() -> None:
    unmasked = {f"c{i}": 10.0 * i for i in range(8)}
    masked = {k: v + 8.0 for k, v in unmasked.items()}  # perfect corr, shift == 8.0
    block = ig.name_mask_gate(masked, unmasked)
    assert block["mean_abs_shift"] == pytest.approx(8.0)
    assert block["rank_corr"] == pytest.approx(1.0)
    assert block["name_mask_pass"] is True
    over = {k: v + 8.01 for k, v in unmasked.items()}
    assert ig.name_mask_gate(over, unmasked)["name_mask_pass"] is False


def test_name_mask_gate_rank_inversion_fails() -> None:
    unmasked = {f"c{i}": 10.0 * i for i in range(8)}
    masked = {f"c{i}": 10.0 * (7 - i) for i in range(8)}  # reversed ranks
    block = ig.name_mask_gate(masked, unmasked)
    assert block["rank_corr"] == pytest.approx(-1.0)
    assert block["name_mask_pass"] is False


def test_name_mask_gate_refuses_mismatched_sets() -> None:
    with pytest.raises(AssertionError):
        ig.name_mask_gate({"a": 1.0, "b": 2.0}, {"a": 1.0, "c": 2.0})


def test_band_agreement_gate_boundaries() -> None:
    bands = {"h": "A", "w": "B", "d": "C", "v": "D"}
    aligned = fz.band_agreement_gate({"h": 90.0, "w": 70.0, "d": 50.0, "v": 10.0}, bands)
    assert aligned["band_agreement_rho"] == pytest.approx(1.0)
    assert aligned["band_agreement_pass"] is True
    inverted = fz.band_agreement_gate({"h": 10.0, "w": 50.0, "d": 70.0, "v": 90.0}, bands)
    assert inverted["band_agreement_rho"] == pytest.approx(-1.0)
    assert inverted["band_agreement_pass"] is False


def test_axis_range_gate_boundary() -> None:
    assert fz.axis_range_gate({"a": 10.0, "b": 18.0})["axis_range_pass"] is True  # == 8.0
    assert fz.axis_range_gate({"a": 10.0, "b": 17.99})["axis_range_pass"] is False


# ---------------------------------------------------------------------------
# (b) masking correctness
# ---------------------------------------------------------------------------
def test_mask_word_boundary_and_case_preservation() -> None:
    masked, n = ig.mask_character_name("Iris said hi. Then iris left, asked IRIS twice.", "Iris")
    assert n == 3
    assert masked == ("The character said hi. Then the character left, asked the character twice.")
    # Word boundary: a name embedded in a longer word is untouched.
    untouched, n2 = ig.mask_character_name("Irises are flowers near Paris.", "Iris")
    assert n2 == 0 and untouched == "Irises are flowers near Paris."


def test_mask_sentence_initial_after_punctuation_and_quotes() -> None:
    masked, n = ig.mask_character_name('Done. "Iris nodded." So Iris won?', "Iris")
    assert n == 2
    assert masked == 'Done. "The character nodded." So the character won?'


def test_mask_possessive() -> None:
    masked, n = ig.mask_character_name("He took Iris's hat.", "Iris")
    assert n == 1
    assert masked == "He took the character's hat."


def test_mask_no_cross_name_collisions_real_panel() -> None:
    # Masking character A's name never touches character B's name — over the
    # REAL 16 panel names (pairwise non-substring by panel constraint).
    for a in NAMES:
        for b in NAMES:
            if a == b:
                continue
            text = f"Later {b} smiled at the crowd."
            masked, n = ig.mask_character_name(text, a)
            assert n == 0 and masked == text, (a, b)
    for a in NAMES:
        masked, n = ig.mask_character_name(f"Later {a} smiled.", a)
        assert n == 1 and a not in masked, a


# ---------------------------------------------------------------------------
# (c) drop-accounting propagation through --step gates (fixtures)
# ---------------------------------------------------------------------------
FIXTURE_DROPS = {
    "n_dropped_draws_content": 3,
    "n_refusal_draws": 1,
    "n_transport_lost_draws": 2,
    "n_total_draws": 200,
}


def _report(tag: str, mean: float, n: int) -> dict:
    return {
        "leg": jl.LEG_AI_LIKENESS,
        "tag": tag,
        "spend_executed": True,
        "rubric_sha256": fz.rubric_fingerprint(),
        "n_items": n,
        **FIXTURE_DROPS,
        "means": {
            "pooled": {"n": n, "mean": mean},
            "capped": {"n": 0, "mean": None},
            "natural": {"n": n, "mean": mean},
            "n_unscored_items": 0,
        },
    }


def _raw(tag: str, per_item: dict[str, float | None], n_draws: int = 2) -> dict:
    """A save_raw fixture: item -> per-draw parsed dicts (None = dropped draws)."""
    all_scores: dict[str, dict] = {}
    for cid, score in per_item.items():
        iid = jl.item_id(jl.LEG_AI_LIKENESS, tag, cid)
        for d in range(n_draws):
            parsed = (
                {"reasoning": "x", "score": 105, "stop_reason": "end_turn"}  # out-of-range: DROP
                if score is None
                else {"reasoning": "x", "score": score, "stop_reason": "end_turn"}
            )
            all_scores[f"{iid}__{d:05d}__00"] = parsed
    return {"all_scores": all_scores}


# Per-character unmasked base scores (A high, D low — monotone, corr 1.0).
UNMASKED_BASE = {"helios": 90.0, "iris": 85.0, "cobalt": 80.0, "vera": 75.0}
UNMASKED_BASE.update({"vex": 20.0, "barnaby": 15.0, "zara": 10.0, "mort": 5.0})
FLAT_MEANS = {
    "helios": 80.0,
    "iris": 79.5,
    "wren": 79.0,
    "elena": 78.5,
    "dana": 78.0,
    "gus": 77.5,
    "vex": 80.5,
    "mort": 78.0,
}


@pytest.fixture()
def gates_fixture(tmp_path: Path) -> dict:
    legs = tmp_path / "legs"
    legs.mkdir()
    axis_raw_dir = tmp_path / "axis_raw"
    axis_raw_dir.mkdir()
    freeze_path = tmp_path / "axis_freeze.json"
    freeze_path.write_text(json.dumps({"issue": 2479, "gates": {"axis_range": 20.0}}))

    for r in INSERTED:
        name = r["name"]
        p = legs / f"judge_report_ail_flat_{name}.json"
        p.write_text(json.dumps(_report(f"flat_{name}", FLAT_MEANS[name], 100)))

    for r in EXTREMES:
        name = r["name"]
        base = UNMASKED_BASE[name]
        conv_ids = ["s1", "s2"] + (["s3"] if name == "helios" else [])
        (legs / f"judge_report_ail_mask_{name}.json").write_text(
            json.dumps(_report(f"mask_{name}", base - 2.0, len(conv_ids)))
        )
        (legs / f"judge_sample_ail_mask_{name}.json").write_text(
            json.dumps({"leg": jl.LEG_AI_LIKENESS, "tag": f"mask_{name}", "conv_ids": conv_ids})
        )
        # Masked raw: base-2 per item; helios s3 drops EVERY draw (out-of-range).
        masked_scores: dict[str, float | None] = {cid: base - 2.0 for cid in conv_ids}
        if name == "helios":
            masked_scores["s3"] = None
        (legs / f"judge_raw_ail_mask_{name}.json").write_text(
            json.dumps(_raw(f"mask_{name}", masked_scores))
        )
        # Unmasked axis raw: base for every sampled item.
        (axis_raw_dir / f"judge_raw_ail_{name}.json").write_text(
            json.dumps(_raw(name, {cid: base for cid in conv_ids}))
        )

    return {
        "legs": legs,
        "freeze": freeze_path,
        "axis_raw_glob": str(axis_raw_dir / "judge_raw_ail_{name}.json"),
        "out": tmp_path / "instrument_gates.json",
    }


def test_step_gates_end_to_end_and_drop_propagation(gates_fixture: dict) -> None:
    payload = ig.step_gates(
        PANEL,
        gates_fixture["freeze"],
        gates_fixture["legs"],
        gates_fixture["axis_raw_glob"],
        gates_fixture["out"],
    )
    assert gates_fixture["out"].is_file()
    on_disk = json.loads(gates_fixture["out"].read_text())
    assert on_disk["gates"] == payload["gates"]

    flat = payload["verbatim_flatness"]
    assert flat["spread"] == pytest.approx(3.0)  # 80.5 - 77.5
    assert flat["threshold"] == pytest.approx(10.0)  # 0.5 * 20.0
    assert flat["verbatim_flatness_pass"] is True
    # Drop accounting propagated verbatim from the leg reports.
    for r in INSERTED:
        drops = flat["per_char"][r["name"]]["drops"]
        for k, v in FIXTURE_DROPS.items():
            assert drops[k] == v, (r["name"], k)

    mask = payload["name_mask"]
    assert mask["mean_abs_shift"] == pytest.approx(2.0)
    assert mask["rank_corr"] == pytest.approx(1.0)
    assert mask["name_mask_pass"] is True
    # helios s3 dropped ALL masked draws: excluded from the masked mean,
    # counted in the per-item accounting, while its unmasked mean survives.
    h = mask["per_char"]["helios"]
    assert h["n_sampled"] == 3
    assert h["n_masked_scored"] == 2
    assert h["n_masked_all_draws_dropped"] == 1
    assert h["n_unmasked_scored"] == 3
    assert mask["per_char_masked_mean"]["helios"] == pytest.approx(88.0)
    assert mask["per_char_unmasked_mean"]["helios"] == pytest.approx(90.0)


def test_step_gates_fails_loud_on_dry_run_report(gates_fixture: dict) -> None:
    # A dry-run leg report (spend_executed False) must never enter the gates.
    name = INSERTED[0]["name"]
    p = gates_fixture["legs"] / f"judge_report_ail_flat_{name}.json"
    report = json.loads(p.read_text())
    report["spend_executed"] = False
    p.write_text(json.dumps(report))
    with pytest.raises(AssertionError, match="spend_executed"):
        ig.step_gates(
            PANEL,
            gates_fixture["freeze"],
            gates_fixture["legs"],
            gates_fixture["axis_raw_glob"],
            gates_fixture["out"],
        )


def test_load_leg_report_missing_file_fails_loud(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="judge-leg report"):
        fz.load_leg_report(tmp_path, "iris")


def test_freeze_rubric_fingerprint_matches_parent_instrument() -> None:
    import hashlib

    assert fz.rubric_fingerprint() == hashlib.sha256(jl.AI_LIKENESS_RUBRIC.encode()).hexdigest()
