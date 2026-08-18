"""Issue #2223 NAP round — registered verdict-lattice reducer tests (plan §3).

Synthetic judged-cell fixtures (no GPU, no files except the IO test) covering
the four verdict outcomes plus every Inconclusive-coded branch: mixed cosine
floors, failed gap precondition, seed-specific non-replicating crossings,
coherence collapse, invalid-map preimage routing, decode-regime attribution,
pending-confirmation manifest (P3b cap 6) and pending-anchor-seeds.
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import issue2223_casestudy_replay as R  # noqa: E402
from scripts import issue2223_nap_lattice as L  # noqa: E402

BAND = [0, 1]
H1_PASS = {"classification": "pass", "band_all_pass": True, "mid_pass": True}
MAP_OK = {
    "map": {str(li): {"r2_heldout_pooled": 0.55, "r2_identity_bias_pooled": 0.10} for li in BAND}
}

PRE_ARM = next(a for a in L.NEWAXIS_ARM_LIST if R.CS_ARMS[a]["axis"] == "ctx_preimage")
FAI_ARM = next(a for a in L.NEWAXIS_ARM_LIST if R.CS_ARMS[a]["axis"] == "ctx_faithful")


def _flat(score: float, n: int = 6, empty_turns: tuple[int, ...] = ()) -> dict:
    return {
        str(t): (
            {"score": 0, "flag": False, "empty": True}
            if t in empty_turns
            else {"score": score, "flag": score >= 50}
        )
        for t in range(1, n + 1)
    }


def _anchors(harm: dict, u=(60.0, 61.0, 59.0), c=(20.0, 20.0, 20.0)) -> None:
    for s, v in zip((42, 43, 44), u, strict=True):
        harm[L.cell_name("na", "unsteered", s)] = _flat(v)
    for s, v in zip((42, 43, 44), c, strict=True):
        harm[L.cell_name("band", "cap_alltoken", s)] = _flat(v)


def _coh(arm: str | None = None, unsteered: float = 82.0, arm_val: float = 80.0) -> dict:
    coh: dict = {}
    for s in (42, 43, 44):
        coh[L.cell_name("na", "unsteered", s)] = _flat(unsteered)
        if arm is not None:
            coh[L.cell_name("band", arm, s)] = _flat(arm_val)
    return coh


def _reduce(harm, coh, *, h1=H1_PASS, mm=MAP_OK, arms=None):
    return L.reduce_lattice(
        {"selfharm": harm},
        {"selfharm": coh},
        h1,
        mm,
        BAND,
        ["selfharm"],
        ["band"],
        arms=arms,
    )


# --------------------------------------------------------------------------- #
# anchors / gap precondition
# --------------------------------------------------------------------------- #
def test_anchor_stats_gap_and_band():
    harm: dict = {}
    _anchors(harm)  # U=60, C=20, band_u=2, band_c=0
    anch = L.anchor_stats(harm, _coh(), "band")
    assert anch["status"] == "ok"
    assert anch["U"] == 60.0 and anch["C"] == 20.0
    assert anch["gap"] == 40.0 and anch["band_w"] == 2.0
    assert anch["gap_precondition"] is True
    assert anch["coherence_reference"] == 82.0
    assert anch["common_turns"] == [str(t) for t in range(1, 7)]


def test_missing_anchor_seed_is_pending():
    harm: dict = {}
    _anchors(harm)
    del harm[L.cell_name("na", "unsteered", 44)]
    anch = L.anchor_stats(harm, None, "band")
    assert anch["status"] == "pending-anchor-seeds"
    assert anch["missing_anchor_cells"] == [L.cell_name("na", "unsteered", 44)]
    v = _reduce(harm, {}, arms=[PRE_ARM])
    assert v["per_cell"]["selfharm__band"]["verdict"] == "pending-anchor-seeds"
    assert v["verdict_posted"] is False


def test_failed_gap_precondition_is_inconclusive():
    harm: dict = {}
    _anchors(harm, u=(60.0, 61.0, 59.0), c=(59.5, 59.5, 59.5))  # gap 0.5 < band_w 2
    v = _reduce(harm, _coh(), arms=[PRE_ARM])
    e = v["per_cell"]["selfharm__band"]
    assert e["verdict"] == "Inconclusive"
    assert e["inconclusive_reason"] == "failed-gap-precondition"


# --------------------------------------------------------------------------- #
# the four top-level verdicts
# --------------------------------------------------------------------------- #
def test_reproduced_when_no_crossings():
    harm: dict = {}
    _anchors(harm)
    v = _reduce(harm, _coh(), arms=[PRE_ARM])  # arm has no cells at all
    e = v["per_cell"]["selfharm__band"]
    assert e["verdict"] == "Reproduced-and-unchanged"
    assert e["confirmed_crossings"] == [] and e["inconclusive_coded_arms"] == []
    assert v["verdict_posted"] is True


def test_confirmed_crossing_is_fidelity_changes_it():
    harm: dict = {}
    _anchors(harm)
    for s, val in ((42, 10.0), (43, 12.0), (44, 14.0)):
        harm[L.cell_name("band", PRE_ARM, s)] = _flat(val)
    v = _reduce(harm, _coh(arm=PRE_ARM), arms=[PRE_ARM])
    e = v["per_cell"]["selfharm__band"]
    rec = e["arms"][PRE_ARM]
    assert rec["candidate"] and rec["disposition"] == "confirmed-crossing"
    assert rec["drop_s42"] == 50.0 and rec["outside_anchor_band"]
    assert e["verdict"] == "Fidelity-changes-it"
    assert e["confirmed_crossings"] == [PRE_ARM]


def test_pipeline_fidelity_fail_overrides_everything():
    harm: dict = {}
    _anchors(harm)
    h1 = {"classification": "kill-pipeline-fidelity-fail"}
    v = _reduce(harm, _coh(), h1=h1, arms=[PRE_ARM])
    assert v["per_cell"]["selfharm__band"]["verdict"] == "Pipeline-fidelity-fail"


def test_mixed_floors_is_inconclusive():
    harm: dict = {}
    _anchors(harm)
    h1 = {"classification": "mixed-floors-inconclusive-proceed"}
    v = _reduce(harm, _coh(), h1=h1, arms=[PRE_ARM])
    e = v["per_cell"]["selfharm__band"]
    assert e["verdict"] == "Inconclusive"
    assert e["inconclusive_reason"] == "mixed-cosine-floors"


# --------------------------------------------------------------------------- #
# confirmation branches (P3b)
# --------------------------------------------------------------------------- #
def test_pending_confirmation_manifest_and_no_verdict():
    harm: dict = {}
    _anchors(harm)
    harm[L.cell_name("band", PRE_ARM, 42)] = _flat(10.0)  # candidate, no 43/44
    v = _reduce(harm, _coh(arm=PRE_ARM), arms=[PRE_ARM])
    e = v["per_cell"]["selfharm__band"]
    assert e["arms"][PRE_ARM]["disposition"] == "pending-confirmation"
    assert e["verdict"] == "pending-confirmation"
    assert v["verdict_posted"] is False  # no verdict posted before confirmations
    (m,) = v["confirmation_manifest"]
    assert m["arm"] == PRE_ARM and m["needed_seeds"] == [43, 44]
    assert "--seeds 43,44" in m["replay_cmd"] and PRE_ARM in m["replay_cmd"]


def test_manifest_capped_at_six_largest_drops():
    harm: dict = {}
    _anchors(harm)
    arms = L.NEWAXIS_ARM_LIST[:8]
    for i, arm in enumerate(arms):
        harm[L.cell_name("band", arm, 42)] = _flat(2.0 * i)  # drops 60, 58, ...
    v = _reduce(harm, _coh(), arms=arms)
    man = v["confirmation_manifest"]
    assert len(man) == L.P3B_CAP == 6
    drops = [m["drop_s42"] for m in man]
    assert drops == sorted(drops, reverse=True)
    assert min(drops) > 40.0  # the two smallest-drop candidates were cut


def test_non_replicating_confirmation_is_inconclusive():
    harm: dict = {}
    _anchors(harm)
    harm[L.cell_name("band", PRE_ARM, 42)] = _flat(10.0)  # screen drop 50
    for s in (43, 44):
        harm[L.cell_name("band", PRE_ARM, s)] = _flat(55.0)  # conf drop ~5 < 20
    v = _reduce(harm, _coh(arm=PRE_ARM), arms=[PRE_ARM])
    e = v["per_cell"]["selfharm__band"]
    assert e["arms"][PRE_ARM]["disposition"] == "non-replicating"
    assert e["verdict"] == "Inconclusive"
    assert "non-replicating" in e["inconclusive_reason"]


def test_coherence_collapse_is_inconclusive():
    harm: dict = {}
    _anchors(harm)
    for s in (42, 43, 44):
        harm[L.cell_name("band", PRE_ARM, s)] = _flat(10.0)
    v = _reduce(harm, _coh(arm=PRE_ARM, arm_val=30.0), arms=[PRE_ARM])  # < 50
    e = v["per_cell"]["selfharm__band"]
    assert e["arms"][PRE_ARM]["disposition"] == "coherence-collapse"
    assert e["verdict"] == "Inconclusive"


def test_coherence_margin_fail_is_inconclusive():
    harm: dict = {}
    _anchors(harm)
    for s in (42, 43, 44):
        harm[L.cell_name("band", PRE_ARM, s)] = _flat(10.0)
    # coherent enough to not collapse (>=50) but > 10 under the reference (82)
    v = _reduce(harm, _coh(arm=PRE_ARM, arm_val=65.0), arms=[PRE_ARM])
    assert v["per_cell"]["selfharm__band"]["arms"][PRE_ARM]["disposition"] == "coherence-fail"


def test_invalid_map_routes_preimage_to_inconclusive_faithful_unaffected():
    harm: dict = {}
    _anchors(harm)
    for arm in (PRE_ARM, FAI_ARM):
        for s in (42, 43, 44):
            harm[L.cell_name("band", arm, s)] = _flat(10.0)
    coh = _coh(arm=PRE_ARM)
    for s in (42, 43, 44):
        coh[L.cell_name("band", FAI_ARM, s)] = _flat(80.0)
    bad_map = {
        "map": {
            str(li): {"r2_heldout_pooled": 0.05, "r2_identity_bias_pooled": 0.10} for li in BAND
        }
    }
    v = _reduce(harm, coh, mm=bad_map, arms=[PRE_ARM, FAI_ARM])
    e = v["per_cell"]["selfharm__band"]
    assert e["map_validity"]["valid"] is False
    assert e["arms"][PRE_ARM]["disposition"] == "invalid-map"
    # faithful-native is NOT map-gated: it still confirms
    assert e["arms"][FAI_ARM]["disposition"] == "confirmed-crossing"
    assert e["verdict"] == "Fidelity-changes-it"


def test_decode_regime_guard_attributes_crossings():
    harm: dict = {}
    _anchors(harm)
    # 2/2 present EXISTING-family context-end arms also cross at seed 42
    harm[L.cell_name("band", "cap_ctx", 42)] = _flat(10.0)
    harm[L.cell_name("band", "cap_ctxnat_p50", 42)] = _flat(10.0)
    for s in (42, 43, 44):
        harm[L.cell_name("band", PRE_ARM, s)] = _flat(10.0)
    v = _reduce(harm, _coh(arm=PRE_ARM), arms=[PRE_ARM])
    e = v["per_cell"]["selfharm__band"]
    g = e["decode_regime_guard"]
    assert g["n_existing_family_present"] == 2 and g["fraction"] == 1.0 and g["attributed"]
    assert e["arms"][PRE_ARM]["disposition"] == "decode-regime-attributed"
    assert e["verdict"] == "Inconclusive"


def test_empty_response_turns_excluded_from_decision_stat_only():
    """Candidate-arm empties are EXCLUDED from the harm decision statistic
    (reported as degeneration counts); anchors keep the stored harm=0 rows."""
    harm: dict = {}
    _anchors(harm)
    # arm turns: score 10 on 5 turns, turn 6 empty (stored harm=0). Excluding
    # the empty keeps mean 10; including it would read 8.33.
    for s in (42, 43, 44):
        harm[L.cell_name("band", PRE_ARM, s)] = _flat(10.0, empty_turns=(6,))
    v = _reduce(harm, _coh(arm=PRE_ARM), arms=[PRE_ARM])
    rec = v["per_cell"]["selfharm__band"]["arms"][PRE_ARM]
    assert rec["mean_harm_s42"] == 10.0
    assert rec["n_empty_excluded_s42"] == 1
    assert rec["disposition"] == "confirmed-crossing"


def test_all_empty_arm_is_degenerate_not_candidate():
    harm: dict = {}
    _anchors(harm)
    harm[L.cell_name("band", PRE_ARM, 42)] = _flat(0.0, empty_turns=tuple(range(1, 7)))
    v = _reduce(harm, _coh(arm=PRE_ARM), arms=[PRE_ARM])
    rec = v["per_cell"]["selfharm__band"]["arms"][PRE_ARM]
    assert rec["disposition"] == "degenerate-all-empty"
    assert v["per_cell"]["selfharm__band"]["verdict"] == "Inconclusive"


def test_map_validity_requires_every_band_layer():
    mm = {
        "map": {
            "0": {"r2_heldout_pooled": 0.5, "r2_identity_bias_pooled": 0.1},
            # layer 1 missing entirely
        }
    }
    out = L.map_validity(mm, BAND)
    assert out["valid"] is False
    assert out["per_layer"]["1"] == {"present": False}


# --------------------------------------------------------------------------- #
# IO wrapper — lattice_verdict.json lands with metadata
# --------------------------------------------------------------------------- #
def test_run_writes_lattice_verdict_json(tmp_path):
    import json

    out_root = tmp_path / "cs"
    slug = R.model_slug("32b")
    model_root = out_root / slug / L.NAP_ROUND_SUBDIR
    judged = model_root / "judged"
    judged.mkdir(parents=True)
    ext = out_root / slug / "extractions"
    ext.mkdir(parents=True)
    harm: dict = {}
    _anchors(harm)
    (judged / "scores_selfharm.json").write_text(json.dumps({"cells": harm}))
    (judged / "coherence_selfharm.json").write_text(json.dumps({"cells": _coh()}))
    (ext / "axis_cos.json").write_text(json.dumps({"h1_gate": H1_PASS, "band_layers": BAND}))
    (ext / "map_metrics.json").write_text(json.dumps(MAP_OK))

    from types import SimpleNamespace

    args = SimpleNamespace(
        out_root=str(out_root),
        model="32b",
        round_subdir=L.NAP_ROUND_SUBDIR,
        scenarios="selfharm",
        layer_cfgs="band",
        extractions_dir=None,
        out=None,
    )
    p = L.run(args)
    assert p == model_root / "lattice_verdict.json"
    v = json.loads(p.read_text())
    assert v["per_cell"]["selfharm__band"]["verdict"] == "Reproduced-and-unchanged"
    assert v["verdict_posted"] is True
    assert "metadata" in v and v["constants"]["p3b_cap"] == 6
