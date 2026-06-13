# marker token + em-dash intentional
"""CPU-only tests for task #632 — the assistant-proximal negative swap (plan §4.3).

#632 forks the #610 mercenary no-default chassis and swaps its 4th negative
slot from ``journalist`` (far from the assistant centroid) to ``programmer``
(the deterministic min-distance proximal pick). The comparator is the #610
no-default arm's 3 committed trajectories, REUSED read-only. Centering drops
``programmer`` (now a trained negative) → a 34-persona set on BOTH arms.

Covers plan §4.3 assertions (a)-(i):
  (a) the ``assistant_proximal`` chassis builds a spec with the proximal panel;
  (b) ``qwen_default`` count == 0 (the inverted parent invariant);
  (c) ``panel ∩ targets == ∅`` and programmer ∉ {ctrl, near} slots;
  (d) the mercenary + software_engineer specs are UNCHANGED + their new fields
      carry the #610-preserving defaults (byte-equivalence regression);
  (e) ``centering_set(assistant_proximal)`` has 34 personas, excludes
      programmer, and the replacement-in-out guard does NOT fire (Diff 3a
      ordering);
  (f) the comparator trajectories on the 34-set reproduce qwen_default −0.1977
      ± 1e-3 AND assistant −0.2063 ± 1e-3 (the recomputed comparison anchors);
  (g) Must-Fix 1: ``comparator_sweep_root == eval_results/issue_610/sweep`` AND
      ``analyze_610`` loads ``with_arm`` from it (not from ``parent_sweep``);
  (h) Must-Fix 2: ``band_verdict`` classifies HELD/PARTIAL/FALSIFIED
      symmetrically on synthetic deltas 0.02 / 0.05 / 0.08 AND the assistant
      comparator anchor is −0.2063 (NOT qwen_default's −0.1977);
  (i) Must-Fix 3: with ``replacement_ctrl_precedent=None`` the analyze run does
      NOT raise; ``replacement_read["passes"] is None``; ``any_miss`` does not
      count it as a miss.

Tests (a)-(e)/(h) are pure-logic / synthetic and always run. Tests (f)-(g)/(i)
read the committed comparator trajectories + the parent manifest and SKIP when
those are not checked out (the sparse-cone case), mirroring
``tests/test_issue610_spec.py``. Runs in <5 s on CPU; no model/tokenizer load.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.experiments.default_dose_610 import (
    CHASSES,
    EXTRA_EVAL_PERSONAS,
    chassis_for_slug,
)
from explore_persona_space.experiments.default_dose_610.analyze import (
    TERMINAL_FRAC,
    band_verdict,
    centered_shift,
    centering_set,
)
from explore_persona_space.experiments.default_dose_610.cells import build_610_spec

SOURCE = "villain"
PARENT_MANIFEST = Path("eval_results/issue_600/panel_selection.json")
COMPARATOR_SWEEP = Path("eval_results/issue_610/sweep")
COMPARATOR_SLUG = "c610_mercenary_near_nodefault"
SEEDS = (42, 137, 219)

# Plan-time recomputed anchors (verified at implementation time by running the
# analyze.py formula against the committed comparator trajectories on the
# 34-set; see plan §4.3 / §6.1). These are the comparison anchors the headline
# band test uses — NEVER the published 35-set −0.1948.
QWEN_DEFAULT_COMPARATOR_34 = -0.1977
ASSISTANT_COMPARATOR_34 = -0.2063


def _manifest_two_chassis() -> dict:
    """Synthetic parent manifest carrying the mercenary chassis shape + a
    distinct second target, with programmer in the held-out (untrained) panel
    so the proximal swap + the 34-set exclusion can be exercised without the
    real #600 manifest."""
    return {
        "schema_version": "i600_panel_selection_v1",
        "bank_content_hash": "f" * 64,
        "base_panel": [{"name": "bartender"}, {"name": "french_person"}],
        "targets": [
            {
                "name": "mercenary",
                "stratum": "near",
                "near": {"name": "dictator"},
                "ctrl": {"name": "journalist"},
            },
            {
                "name": "pirate_captain",
                "stratum": "near",
                "near": {"name": "dictator"},
                "ctrl": {"name": "journalist"},
            },
        ],
        # held_out_panel includes programmer (a #610-untrained persona) + two
        # filler untrained personas so the 35→34 exclusion arithmetic in
        # centering_set has something to drop. (The synthetic EXPECTED_CENTERING_N
        # is patched per-test where the real 35-count matters.)
        "held_out_panel": ["mercenary", "pirate_captain", "programmer", "p1", "p2"],
        "q_eval": ["q?"],
    }


# ── (a)-(c): the proximal spec builds + the inverted/disjointness invariants. ─


def test_a_assistant_proximal_spec_panel():
    """(a) the assistant_proximal chassis builds the proximal panel."""
    spec = build_610_spec(_manifest_two_chassis(), CHASSES["assistant_proximal"])
    assert spec.slug == "c632_assistant_proximal"
    assert spec.panel == ("programmer", "bartender", "french_person", "dictator")
    assert spec.target == "mercenary"
    assert spec.slot_persona == "programmer"


def test_b_qwen_default_count_zero():
    """(b) qwen_default appears ZERO times in the proximal panel."""
    spec = build_610_spec(_manifest_two_chassis(), CHASSES["assistant_proximal"])
    assert spec.panel.count("qwen_default") == 0
    assert "qwen_default" not in spec.panel


def test_c_panel_disjoint_from_targets_and_slots():
    """(c) panel ∩ targets == ∅ and programmer ∉ {ctrl, near} slots."""
    m = _manifest_two_chassis()
    spec = build_610_spec(m, CHASSES["assistant_proximal"])
    target_names = {t["name"] for t in m["targets"]}
    assert not (set(spec.panel) & target_names)
    assert SOURCE not in spec.panel
    # programmer is neither the mercenary ctrl (journalist) nor the near (dictator)
    merc = next(t for t in m["targets"] if t["name"] == "mercenary")
    assert "programmer" not in (merc["ctrl"]["name"], merc["near"]["name"])
    assert spec.panel.count("programmer") == 1


def test_c_proximal_replacement_colliding_with_slot_fails():
    """(c, error path) a proximal replacement equal to a parent slot is rejected
    by the new non-ctrl pre-registration check (Diff 2)."""
    import dataclasses

    bad = dataclasses.replace(CHASSES["assistant_proximal"], replacement="dictator")
    with pytest.raises(AssertionError, match="collides with a"):
        build_610_spec(_manifest_two_chassis(), bad)


# ── (d): byte-equivalence regression — the two #610 chassis are UNCHANGED. ────


def test_d_existing_chassis_new_fields_preserve_610():
    """(d) the mercenary + software_engineer chassis carry the #610-preserving
    defaults for the four new fields (so #610 stays byte-equivalent): a None
    comparator_sweep_root (→ falls back to parent_sweep), an empty
    centering_extra_exclude (→ 35-set), replacement_is_ctrl=True, and their
    original float ctrl precedents."""
    for name in ("mercenary", "software_engineer"):
        c = CHASSES[name]
        assert c.comparator_sweep_root is None, name
        assert c.centering_extra_exclude == (), name
        assert c.replacement_is_ctrl is True, name
        assert isinstance(c.replacement_ctrl_precedent, float), name
    assert CHASSES["mercenary"].replacement_ctrl_precedent == -0.117
    assert CHASSES["software_engineer"].replacement_ctrl_precedent == -0.0372


def test_d_existing_chassis_specs_unchanged():
    """(d) the mercenary + software_engineer specs build exactly as #610."""
    m = {
        "schema_version": "i600_panel_selection_v1",
        "bank_content_hash": "f" * 64,
        "base_panel": [{"name": "bartender"}, {"name": "french_person"}],
        "targets": [
            {
                "name": "mercenary",
                "stratum": "near",
                "near": {"name": "dictator"},
                "ctrl": {"name": "journalist"},
            },
            {
                "name": "software_engineer",
                "stratum": "mid",
                "near": {"name": "data_scientist"},
                "ctrl": {"name": "hospice_nurse"},
            },
        ],
        "held_out_panel": ["mercenary", "software_engineer", "p1", "p2"],
        "q_eval": ["q?"],
    }
    merc = build_610_spec(m, CHASSES["mercenary"])
    assert merc.panel == ("journalist", "bartender", "french_person", "dictator")
    se = build_610_spec(m, CHASSES["software_engineer"])
    assert se.panel == ("hospice_nurse", "bartender", "french_person", "data_scientist")


# ── (e)+(f): centering set (34) + recomputed comparator anchors. ─────────────


@pytest.mark.skipif(not PARENT_MANIFEST.exists(), reason="parent manifest not checked out")
def test_e_centering_set_34_excludes_programmer():
    """(e) centering_set(assistant_proximal) has 34 personas, excludes
    programmer (the trained replacement), and the replacement-in-out guard does
    NOT fire (Diff 3a ordering: extra-exclude applied BEFORE that guard)."""
    manifest = json.loads(PARENT_MANIFEST.read_text())
    cs_merc = centering_set(manifest, CHASSES["mercenary"])
    cs_prox = centering_set(manifest, CHASSES["assistant_proximal"])
    assert len(cs_merc) == 35  # the #610 set is unchanged
    assert "programmer" in cs_merc  # programmer survives trained_anywhere in #610
    assert len(cs_prox) == 34  # #632 drops programmer
    assert "programmer" not in cs_prox
    assert SOURCE not in cs_prox
    # 34-set is exactly the 35-set minus programmer (no other personas moved).
    assert set(cs_prox) == set(cs_merc) - {"programmer"}


@pytest.mark.skipif(
    not (PARENT_MANIFEST.exists() and (COMPARATOR_SWEEP / COMPARATOR_SLUG).is_dir()),
    reason="parent manifest or #610 comparator trajectories not checked out",
)
def test_f_comparator_anchors_recompute_on_34_set():
    """(f) the #610 no-default comparator on the 34-set reproduces qwen_default
    −0.1977 ± 1e-3 AND assistant −0.2063 ± 1e-3 (the recomputed comparison
    anchors, NOT the published 35-set −0.1948)."""
    manifest = json.loads(PARENT_MANIFEST.read_text())
    centering = centering_set(manifest, CHASSES["assistant_proximal"])  # 34-set
    arms = {
        s: json.loads(
            (COMPARATOR_SWEEP / COMPARATOR_SLUG / f"seed_{s}" / "trajectory.json").read_text()
        )
        for s in SEEDS
    }
    qd = {s: centered_shift(arms[s], TERMINAL_FRAC, "qwen_default", centering) for s in SEEDS}
    asst = {s: centered_shift(arms[s], TERMINAL_FRAC, "assistant", centering) for s in SEEDS}
    assert float(np.median(list(qd.values()))) == pytest.approx(
        QWEN_DEFAULT_COMPARATOR_34, abs=1e-3
    )
    assert float(np.median(list(asst.values()))) == pytest.approx(ASSISTANT_COMPARATOR_34, abs=1e-3)


# ── (g): Must-Fix 1 — comparator root resolution. ────────────────────────────


def test_g_comparator_sweep_root_field():
    """(g) the assistant_proximal chassis points at issue_610/sweep (Must-Fix 1)."""
    assert CHASSES["assistant_proximal"].comparator_sweep_root == COMPARATOR_SWEEP


@pytest.mark.skipif(
    not (PARENT_MANIFEST.exists() and (COMPARATOR_SWEEP / COMPARATOR_SLUG).is_dir()),
    reason="parent manifest or #610 comparator trajectories not checked out",
)
def test_g_analyze_loads_with_arm_from_comparator_root(tmp_path: Path, monkeypatch):
    """(g) analyze_610 resolves the comparator from comparator_sweep_root, NOT
    from parent_sweep: pointing parent_sweep at an EMPTY dir still finds the
    comparator (because the chassis comparator_sweep_root carries it)."""
    import explore_persona_space.experiments.default_dose_610.analyze as az

    chassis = CHASSES["assistant_proximal"]
    loaded_from: dict[str, Path] = {}
    real_load_arm = az.load_arm

    def _spy_load_arm(sweep_dir, slug, seeds, required_personas):
        loaded_from[slug] = sweep_dir
        return real_load_arm(sweep_dir, slug, seeds, required_personas)

    monkeypatch.setattr(az, "load_arm", _spy_load_arm)
    # Stop the run right after both arms load (before the new arm's eval-only
    # personas are needed): point new_sweep at the comparator so load_arm
    # succeeds for both, then assert the comparator root was used for with_arm.
    empty_parent = tmp_path / "empty_parent_sweep"
    empty_parent.mkdir()
    with pytest.raises((AssertionError, FileNotFoundError, KeyError)):
        # the new arm requires qwen_default/assistant which the comparator
        # trajectories DO carry, so this gets far enough to also load without_arm;
        # whichever read fails first, with_arm has already been recorded.
        az.analyze_610(
            parent_sweep=empty_parent,
            new_sweep=COMPARATOR_SWEEP,
            manifest_path=PARENT_MANIFEST,
            out_path=tmp_path / "analysis.json",
            figures_dir=tmp_path / "figs",
            seeds=SEEDS,
            chassis=chassis,
        )
    # with_arm (the comparator slug) was loaded from the chassis comparator
    # root, NOT from the empty parent_sweep.
    assert loaded_from[COMPARATOR_SLUG] == COMPARATOR_SWEEP


# ── (h): Must-Fix 2 — band_verdict symmetry + the assistant anchor. ──────────


@pytest.mark.parametrize(
    "delta, expected",
    [
        (0.02, "HELD"),  # |Δ| ≤ band (0.033)
        (-0.02, "HELD"),  # symmetric
        (0.05, "PARTIAL"),  # band < |Δ| ≤ 2*band (0.066)
        (-0.05, "PARTIAL"),
        (0.08, "FALSIFIED"),  # |Δ| > 2*band
        (-0.08, "FALSIFIED"),
    ],
)
def test_h_band_verdict_symmetric(delta, expected):
    """(h) band_verdict classifies HELD/PARTIAL/FALSIFIED symmetrically."""
    v = band_verdict(0.0 + delta, 0.0, 0.033)
    assert v["verdict"] == expected
    assert v["abs_delta"] == pytest.approx(abs(delta))
    assert v["direction"] == ("less_shielded" if delta > 0 else "more_shielded")


def test_h_band_verdict_boundaries():
    """(h) exact boundary behavior: |Δ| == band is HELD, |Δ| == 2*band is
    PARTIAL, strictly above 2*band is FALSIFIED."""
    assert band_verdict(0.033, 0.0, 0.033)["verdict"] == "HELD"
    assert band_verdict(0.066, 0.0, 0.033)["verdict"] == "PARTIAL"
    assert band_verdict(0.0661, 0.0, 0.033)["verdict"] == "FALSIFIED"


@pytest.mark.skipif(
    not (PARENT_MANIFEST.exists() and (COMPARATOR_SWEEP / COMPARATOR_SLUG).is_dir()),
    reason="parent manifest or #610 comparator trajectories not checked out",
)
def test_h_assistant_band_anchored_on_assistant_comparator(tmp_path: Path):
    """(h) the assistant secondary verdict is anchored on the assistant
    comparator's OWN median (−0.2063), NOT qwen_default's (−0.1977). Runs the
    real analyze against the comparator as BOTH arms (a degenerate but valid
    self-comparison) and reads secondary['assistant_band_verdict']."""
    chassis = CHASSES["assistant_proximal"]
    result = _run_analyze_self_comparison(chassis, tmp_path)
    sec = result["secondary"]
    assert sec["assistant_band_verdict"]["median_comparator"] == pytest.approx(
        ASSISTANT_COMPARATOR_34, abs=1e-3
    )
    # NOT the qwen_default anchor.
    assert sec["assistant_band_verdict"]["median_comparator"] != pytest.approx(
        QWEN_DEFAULT_COMPARATOR_34, abs=1e-4
    )
    # qwen_default headline verdict carries its own anchor.
    assert result["headline"]["band_verdict_qwen_default"]["median_comparator"] == pytest.approx(
        QWEN_DEFAULT_COMPARATOR_34, abs=1e-3
    )


# ── (i): Must-Fix 3 — None ctrl precedent is skipped, not failed. ────────────


@pytest.mark.skipif(
    not (PARENT_MANIFEST.exists() and (COMPARATOR_SWEEP / COMPARATOR_SLUG).is_dir()),
    reason="parent manifest or #610 comparator trajectories not checked out",
)
def test_i_none_ctrl_precedent_skipped_not_failed(tmp_path: Path):
    """(i) with replacement_ctrl_precedent=None the analyze run does NOT raise;
    replacement_read['passes'] is None (skipped); any_miss does not count it."""
    chassis = CHASSES["assistant_proximal"]
    assert chassis.replacement_ctrl_precedent is None  # precondition
    result = _run_analyze_self_comparison(chassis, tmp_path)
    rr = result["sanity"]["replacement_trained_read"]
    assert rr["persona"] == "programmer"
    assert rr["passes"] is None  # skipped, NOT False
    assert rr["ctrl_precedent"] is None
    # any_miss must not be flipped True purely by the skipped precedent. (The
    # sanity drift-detectors here are a self-comparison so their delta is 0 →
    # they pass; the only None is the precedent, which must not count.)
    assert result["sanity"]["any_miss"] is False


# ── Shared helper: run the real analyze with the comparator as BOTH arms. ────


def _run_analyze_self_comparison(chassis, tmp_path: Path) -> dict:
    """Run analyze_610 with the #610 comparator trajectories supplied as BOTH
    the comparator arm AND the new arm (a degenerate self-comparison that
    exercises every read path on real four-float leaves). Both arms then have
    qwen_default + assistant + programmer, so every centered_shift resolves.
    The comparator lives under issue_610/sweep, so we pass it as new_sweep too;
    comparator_sweep_root carries the with_arm load."""
    import explore_persona_space.experiments.default_dose_610.analyze as az

    # The new arm's slug differs from the comparator slug, so symlink/copy the
    # comparator seed dirs under the new slug in a scratch sweep dir.
    new_sweep = tmp_path / "new_sweep"
    src = COMPARATOR_SWEEP / COMPARATOR_SLUG
    dst = new_sweep / chassis.new_slug
    dst.mkdir(parents=True)
    for seed in SEEDS:
        seed_dst = dst / f"seed_{seed}"
        seed_dst.mkdir()
        (seed_dst / "trajectory.json").write_text(
            (src / f"seed_{seed}" / "trajectory.json").read_text()
        )
    # An EMPTY (but existing) parent_sweep so default_specific_gap_median's
    # iterdir() succeeds and finds no #600 mixes (finer-calib → None strip,
    # handled by the headline). The ctrl-precedent path is guarded out by
    # Diff 3b (replacement_ctrl_precedent=None), so parent_sweep is otherwise
    # unused for this chassis.
    empty_parent = tmp_path / "empty_parent_sweep"
    empty_parent.mkdir()
    return az.analyze_610(
        parent_sweep=empty_parent,
        new_sweep=new_sweep,
        manifest_path=PARENT_MANIFEST,
        out_path=tmp_path / "analysis.json",
        figures_dir=tmp_path / "figs",
        seeds=SEEDS,
        chassis=chassis,
    )


# ── extra_eval_personas sanity (the primary DV + cluster probe are wired). ───


def test_extra_eval_personas_unchanged():
    """programmer needs no extra wiring (already in held_out); the extra-eval
    set stays the primary DV + cluster probe."""
    assert EXTRA_EVAL_PERSONAS == ("qwen_default", "assistant")


def test_chassis_for_slug_resolves_assistant_proximal():
    """The dispatcher resolves the new chassis from --cell (Diff 4, no code change)."""
    assert chassis_for_slug("c632_assistant_proximal").name == "assistant_proximal"
