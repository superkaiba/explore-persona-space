# ruff: noqa: RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ΔG + × intentional
"""Task #477 — calibrate.py + 477 module unit tests.

Pure-function, fast, CPU-only. Pins:
  * pick_lr_for_count happy-path (single qualifying LR, multiple qualifying LRs
    → closest-to-target wins);
  * pick_lr_for_count raises (the §7 kill criterion) when no LR satisfies BOTH
    gates;
  * validity_gate (kept/excluded split);
  * the partial-Spearman ≥3-count-level coverage guard (plan §6 discipline #5);
  * the CELL_SPECS_477 registry shape (20 + 8 + 6 = 34 cells total when count
    × LR is properly de-duplicated AT THE REGISTRY level: 4 calib slugs + 4 main
    slugs + 3 implant_sweep slugs = 11 distinct slugs).
"""

from __future__ import annotations

import pytest

# ── Module registry + helper functions ───────────────────────────────────────


def test_cell_specs_477_shape() -> None:
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        ANCHOR_COUNT,
        CALIB_SLUGS,
        CALIBRATION_LR_GRID,
        CELL_SPECS_477,
        COUNT_LEVELS,
        IMPLANT_SWEEP_LRS,
        IMPLANT_SWEEP_SLUGS,
        MAIN_SLUGS,
    )

    # 4 calibration slugs (one per count; LR threaded per-launch, not per-slug)
    # + 4 main slugs + 3 implant_sweep slugs = 11 distinct.
    assert len(CALIB_SLUGS) == len(COUNT_LEVELS) == 4
    assert len(MAIN_SLUGS) == 4
    assert len(IMPLANT_SWEEP_SLUGS) == len(IMPLANT_SWEEP_LRS) == 3
    assert len(CELL_SPECS_477) == 11
    # No duplicates.
    slugs = [c[0] for c in CELL_SPECS_477]
    assert len(set(slugs)) == len(slugs)
    # Anchor count is one of the count levels.
    assert ANCHOR_COUNT in COUNT_LEVELS
    # CALIBRATION_LR_GRID has 5 LRs spanning 25×.
    assert len(CALIBRATION_LR_GRID) == 5
    assert max(CALIBRATION_LR_GRID) / min(CALIBRATION_LR_GRID) >= 20.0


def test_count_for_slug_round_trip() -> None:
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        ANCHOR_COUNT,
        CALIB_SLUGS,
        COUNT_LEVELS,
        IMPLANT_SWEEP_SLUGS,
        MAIN_SLUGS,
        count_for_slug,
        slug_for_count,
    )

    for c in COUNT_LEVELS:
        assert count_for_slug(slug_for_count(c, "calibration")) == c
        assert count_for_slug(slug_for_count(c, "main")) == c
    for slug in CALIB_SLUGS + MAIN_SLUGS:
        assert count_for_slug(slug) in COUNT_LEVELS
    # implant_sweep slugs map to the anchor count.
    for slug in IMPLANT_SWEEP_SLUGS:
        assert count_for_slug(slug) == ANCHOR_COUNT


def test_lr_for_implant_sweep_slug() -> None:
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        IMPLANT_SWEEP_LRS,
        IMPLANT_SWEEP_SLUGS,
        lr_for_implant_sweep_slug,
    )

    for slug, expected_lr in zip(IMPLANT_SWEEP_SLUGS, IMPLANT_SWEEP_LRS, strict=True):
        assert lr_for_implant_sweep_slug(slug) == pytest.approx(expected_lr)
    with pytest.raises(KeyError):
        lr_for_implant_sweep_slug("c477_calib_negp_4")  # not an implant-sweep slug


def test_slug_for_count_rejects_unknown_phase_and_count() -> None:
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import slug_for_count

    with pytest.raises(ValueError):
        slug_for_count(999, "calibration")  # 999 not in COUNT_LEVELS
    with pytest.raises(ValueError):
        slug_for_count(4, "implant_sweep")  # implant_sweep slugs encode LR, not count
    with pytest.raises(ValueError):
        slug_for_count(4, "nonsense_phase")


# ── pick_lr_for_count ────────────────────────────────────────────────────────


def _table(per_count: dict[int, dict[float, dict]]) -> dict[int, dict[float, dict]]:
    """Build a calibration-table fixture from {count: {lr: (delta_g, emit)}}."""
    out: dict[int, dict[float, dict]] = {}
    for cnt, row in per_count.items():
        out[cnt] = {}
        for lr, payload in row.items():
            if isinstance(payload, tuple):
                delta, emit = payload
                out[cnt][lr] = {
                    "source_self_delta_g": delta,
                    "source_emission_p": emit,
                }
            else:
                out[cnt][lr] = payload
    return out


def test_pick_lr_single_qualifying_lr_wins() -> None:
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        pick_lr_for_count,
    )

    # Only lr=1e-5 lands in [10.5, 13.5] with emit≥0.30.
    table = _table(
        {
            4: {
                5e-6: (8.0, 0.50),  # below band
                1e-5: (12.1, 0.50),  # qualifies (delta ok + emit ok)
                2e-5: (15.0, 0.50),  # above band
            }
        }
    )
    pick = pick_lr_for_count(table, 4)
    assert pick["lr"] == pytest.approx(1e-5)
    assert pick["achieved_delta_g"] == pytest.approx(12.1)
    assert pick["in_band"] is True


def test_pick_lr_multiple_qualifying_closest_to_target() -> None:
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        pick_lr_for_count,
    )

    # Both 1e-5 (δ=12.4, dist 0.4) and 2e-5 (δ=11.0, dist 1.0) qualify.
    # 1e-5 wins because closer to TARGET=12.0.
    table = _table(
        {
            8: {
                1e-5: (12.4, 0.50),  # closer to target
                2e-5: (11.0, 0.50),  # farther from target
            }
        }
    )
    pick = pick_lr_for_count(table, 8)
    assert pick["lr"] == pytest.approx(1e-5)


def test_pick_lr_emission_floor_excludes_band_hitter() -> None:
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        pick_lr_for_count,
    )

    # 5e-6 hits band exactly (δ=12.0) but emission is below 0.30 → DISQUALIFIED.
    # 1e-5 misses the implant target slightly but emit OK → qualifies + wins.
    table = _table(
        {
            2: {
                5e-6: (12.0, 0.10),  # band-hitter but sub-emission
                1e-5: (13.0, 0.50),  # delta=13.0 ∈ [10.5, 13.5], emit ≥ 0.30
            }
        }
    )
    pick = pick_lr_for_count(table, 2)
    assert pick["lr"] == pytest.approx(1e-5)


def test_pick_lr_raises_on_kill_criterion() -> None:
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        pick_lr_for_count,
    )

    # No LR satisfies BOTH gates: 5e-6 too low δ, 1e-5 saturated, 2e-5 sub-emit.
    table = _table(
        {
            16: {
                5e-6: (6.0, 0.20),  # below band, sub-emit
                1e-5: (18.0, 1.00),  # above band
                2e-5: (12.0, 0.10),  # band-hit but sub-emit
            }
        }
    )
    with pytest.raises(RuntimeError, match="NO qualifying LR"):
        pick_lr_for_count(table, 16)


def test_pick_lr_accepts_str_count_keys() -> None:
    """JSON deserialization gives str count keys; pick must handle either form."""
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        pick_lr_for_count,
    )

    table = {
        "4": {
            "1e-05": {"source_self_delta_g": 12.0, "source_emission_p": 0.50},
        }
    }
    pick = pick_lr_for_count(table, 4)
    assert pick["lr"] == pytest.approx(1e-5)


def test_pick_lr_raises_on_missing_count() -> None:
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        pick_lr_for_count,
    )

    table = _table({4: {1e-5: (12.0, 0.50)}})
    with pytest.raises(KeyError, match="count=8 missing"):
        pick_lr_for_count(table, 8)


# ── validity_gate ────────────────────────────────────────────────────────────


def test_validity_gate_kept_excluded_split() -> None:
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        validity_gate,
    )

    cells = [
        # KEPT: in band + emit OK.
        {
            "cell": "c1",
            "source_self_delta_g_at_last_ckpt": 12.0,
            "source_emission_p_at_last_ckpt": 0.50,
        },
        # EXCLUDED: below band.
        {
            "cell": "c2",
            "source_self_delta_g_at_last_ckpt": 8.0,
            "source_emission_p_at_last_ckpt": 0.50,
        },
        # EXCLUDED: above band.
        {
            "cell": "c3",
            "source_self_delta_g_at_last_ckpt": 15.0,
            "source_emission_p_at_last_ckpt": 0.80,
        },
        # EXCLUDED: in band but sub-emission.
        {
            "cell": "c4",
            "source_self_delta_g_at_last_ckpt": 11.5,
            "source_emission_p_at_last_ckpt": 0.10,
        },
        # KEPT: edge of band (delta=13.5, emit=0.30 boundary).
        {
            "cell": "c5",
            "source_self_delta_g_at_last_ckpt": 13.5,
            "source_emission_p_at_last_ckpt": 0.30,
        },
    ]
    kept, excluded = validity_gate(cells)
    assert [c["cell"] for c in kept] == ["c1", "c5"]
    assert [c["cell"] for c in excluded] == ["c2", "c3", "c4"]


def test_validity_gate_fails_loud_on_missing_keys() -> None:
    from explore_persona_space.experiments.contrastive_neg_count_decouple_477.calibrate import (
        validity_gate,
    )

    with pytest.raises(KeyError, match="source_self_delta_g_at_last_ckpt"):
        validity_gate([{"cell": "bad", "source_emission_p_at_last_ckpt": 0.5}])
    with pytest.raises(KeyError, match="source_emission_p_at_last_ckpt"):
        validity_gate([{"cell": "bad", "source_self_delta_g_at_last_ckpt": 12.0}])


# ── Partial Spearman ≥3-count-level guard (plan §6 item 5) ──────────────────


def test_partial_spearman_refuses_below_3_count_levels() -> None:
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        partial_spearman_count_given_implant,
    )

    # Only 2 distinct count levels survive — partial Spearman MUST refuse.
    # All cells are phase="main" (H1 partial-spearman defensive assert).
    kept = [
        {
            "cell": "c1",
            "seed": 42,
            "count": 4,
            "lr": 1e-5,
            "phase": "main",
            "source_self_delta_g_at_last_ckpt": 12.0,
            "source_emission_p_at_last_ckpt": 0.5,
            "mean_bystander_delta_g": 5.0,
            "step_at_last_ckpt": 100,
        },
        {
            "cell": "c2",
            "seed": 137,
            "count": 4,
            "lr": 1e-5,
            "phase": "main",
            "source_self_delta_g_at_last_ckpt": 12.1,
            "source_emission_p_at_last_ckpt": 0.5,
            "mean_bystander_delta_g": 5.1,
            "step_at_last_ckpt": 100,
        },
        {
            "cell": "c3",
            "seed": 42,
            "count": 8,
            "lr": 5e-6,
            "phase": "main",
            "source_self_delta_g_at_last_ckpt": 12.5,
            "source_emission_p_at_last_ckpt": 0.5,
            "mean_bystander_delta_g": 6.0,
            "step_at_last_ckpt": 200,
        },
        {
            "cell": "c4",
            "seed": 137,
            "count": 8,
            "lr": 5e-6,
            "phase": "main",
            "source_self_delta_g_at_last_ckpt": 12.4,
            "source_emission_p_at_last_ckpt": 0.5,
            "mean_bystander_delta_g": 6.1,
            "step_at_last_ckpt": 200,
        },
    ]
    out = partial_spearman_count_given_implant(kept, n_bootstrap=50)
    assert out["interpretable"] is False
    assert out["n_count_levels"] == 2
    assert "coverage floor violated" in out["note"]


def test_partial_spearman_computes_at_3_count_levels() -> None:
    """≥3 count levels surviving → partial Spearman IS computed."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        partial_spearman_count_given_implant,
    )

    # 3 count levels × 2 seeds = 6 cells; bystander DV varies a bit with count.
    # All cells are phase="main" (H1 partial-spearman defensive assert).
    kept = []
    for seed in (42, 137):
        for cnt, by_dg in ((2, 4.0), (4, 5.0), (8, 6.0)):
            kept.append(
                {
                    "cell": f"c477_main_calib_negp_{cnt}",
                    "seed": seed,
                    "count": cnt,
                    "lr": 1e-5,
                    "phase": "main",
                    "source_self_delta_g_at_last_ckpt": 12.0 + 0.05 * cnt,
                    "source_emission_p_at_last_ckpt": 0.5,
                    "mean_bystander_delta_g": by_dg + 0.1 * (seed - 42) / 95.0,
                    "step_at_last_ckpt": 100 + 50 * cnt,
                }
            )
    out = partial_spearman_count_given_implant(kept, n_bootstrap=50)
    assert out["interpretable"] is True
    assert out["n"] == 6
    assert out["n_count_levels"] == 3
    # rho_given_implant SHOULD be defined (NaN OK — what matters is the contract).
    assert "rho_given_implant" in out
    assert "bootstrap_given_implant" in out
    assert out["bootstrap_given_implant"]["ci_level"] == pytest.approx(0.90)
    # The robustness partial (vs implant + step) is also reported.
    assert "rho_given_implant_and_step" in out
    assert "bootstrap_given_implant_and_step" in out
    # Per-seed sign is reported for both seeds.
    assert set(out["per_seed_sign"]) == {"42", "137"}


def test_implant_only_axis_spearman_verdict_branches() -> None:
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        implant_only_axis_spearman,
    )

    # Strong positive correlation (ρ near 1.0) — should CONFIRM H2.
    strong_cells = []
    for seed in (42, 137):
        for lr, dv_b, dv_a in ((5e-6, 8.0, 4.0), (1e-5, 12.0, 6.0), (2e-5, 16.0, 8.0)):
            strong_cells.append(
                {
                    "seed": seed,
                    "lr": lr,
                    "source_self_delta_g_at_last_ckpt": dv_b,
                    "mean_bystander_delta_g": dv_a,
                }
            )
    out = implant_only_axis_spearman(strong_cells)
    assert out["n"] == 6
    assert out["verdict"] == "confirms"

    # Near-zero correlation: bystander is constant across implant → ρ=NaN
    # (degenerate), verdict can't be "confirms". Use small dispersion to keep it
    # well-defined but small.
    weak_cells = []
    for seed in (42, 137):
        for lr, dv_b in ((5e-6, 8.0), (1e-5, 12.0), (2e-5, 16.0)):
            weak_cells.append(
                {
                    "seed": seed,
                    "lr": lr,
                    "source_self_delta_g_at_last_ckpt": dv_b,
                    # bystander ≈ constant, tiny noise.
                    "mean_bystander_delta_g": 5.0 + 0.01 * (seed - 42) / 95.0,
                }
            )
    out_weak = implant_only_axis_spearman(weak_cells)
    assert out_weak["verdict"] in ("falsifies", "indeterminate")


# ── Resume-skip pick path (round-2 MAJOR fix) ────────────────────────────────


def test_phase_calibration_pick_handles_resumed_skip_cells(tmp_path):
    """Round-2 MAJOR: resumed-skip cells must carry ΔG/emission keys.

    When `--resume` skips an already-done calibration cell, the dispatcher's
    `_launch` returns None and the scheduler appends a `resumed_skip` result.
    Prior to this fix that dict carried only the unit fields (cell/seed/lr/phase
    + assigned_gpu/status) and `_phase_calibration_pick` KeyErrored on
    `source_self_delta_g_at_last_ckpt`. The fix loads the cell's
    `cell_summary.json` from disk and merges its keys into the appended dict
    (mirroring the `done` path's `**cs` merge).

    This test feeds a `resumed_skip`-shaped result (with the merged ΔG/emission
    keys, as the fix now produces) through `_phase_calibration_pick` and
    asserts (a) no KeyError, (b) the correct pick lands at the qualifying LR.
    """
    import importlib.util
    import sys
    from pathlib import Path

    # Load dispatch_neg_geometry_477 by file path (scripts/ is not a package).
    dispatcher_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "dispatch_neg_geometry_477.py"
    )
    spec = importlib.util.spec_from_file_location("dispatch_neg_geometry_477", dispatcher_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dispatch_neg_geometry_477"] = mod
    spec.loader.exec_module(mod)

    from explore_persona_space.experiments.contrastive_neg_count_decouple_477 import (
        CALIB_SLUGS,
        CALIBRATION_LR_GRID,
        count_for_slug,
    )

    # Build a `resumed_skip` result dict for EACH (calibration slug, LR) cell,
    # carrying the ΔG/emission keys exactly as the fixed _launch path now does
    # (via the `**cs` merge of cell_summary.json). For each count, one LR
    # qualifies (1e-5 lands ΔG=12.0 + emit=0.50); the rest miss the band or
    # emission floor so we exercise the picker's qualification logic too.
    calibration_results: list[dict] = []
    for slug in CALIB_SLUGS:
        cnt = count_for_slug(slug)
        for lr in CALIBRATION_LR_GRID:
            # The qualifying LR for every count is 1e-5 (in-band + emit OK).
            # Stay strictly inside [TARGET_SOURCE_DELTA_G ± MATCH_BAND] =
            # [10.5, 13.5] nats; tiny per-count nudge for differentiation.
            if lr == 1e-5:
                delta_g, emit = 12.0 + 0.05 * (cnt - 4), 0.50  # in-band qualifies
            elif lr < 1e-5:
                delta_g, emit = 6.0, 0.10  # below band, sub-emit
            else:
                delta_g, emit = 17.0, 0.95  # above band
            calibration_results.append(
                {
                    "cell": slug,
                    "seed": 42,
                    "lr": lr,
                    "phase": "calibration",
                    "assigned_gpu": 0,
                    "status": "resumed_skip",
                    # Merged from cell_summary.json (the fix):
                    "run_label": f"{slug}_seed42_lr{lr:g}",
                    "source_self_delta_g_at_last_ckpt": delta_g,
                    "source_emission_p_at_last_ckpt": emit,
                    "mean_bystander_delta_g": 4.0,
                    "step_at_last_ckpt": 100,
                }
            )

    pick_path = tmp_path / "calibration_pick.json"
    # The fix means this call must NOT KeyError on resumed_skip rows.
    picks = mod._phase_calibration_pick(calibration_results, pick_path)
    # Every count level resolves to lr=1e-5.
    assert set(picks.keys()) == {count_for_slug(s) for s in CALIB_SLUGS}
    for cnt, pick in picks.items():
        assert pick["lr"] == pytest.approx(1e-5), f"count={cnt} picked wrong LR: {pick['lr']}"
        assert pick["in_band"] is True


def test_phase_calibration_pick_keyerrors_when_resumed_skip_missing_keys(tmp_path):
    """Regression: an old-shape resumed_skip dict (no ΔG keys) MUST fail loud.

    Pre-fix, _phase_calibration_pick silently KeyErrored without context.
    The fix loads cell_summary.json on the resume branch; this test pins that
    the picker still raises a clear KeyError when handed a malformed row (i.e.
    a row missing the gate metrics) so silent-success regressions can't happen.
    """
    import importlib.util
    import sys
    from pathlib import Path

    dispatcher_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "dispatch_neg_geometry_477.py"
    )
    spec = importlib.util.spec_from_file_location("dispatch_neg_geometry_477", dispatcher_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["dispatch_neg_geometry_477"] = mod
    spec.loader.exec_module(mod)

    bad_results = [
        {
            "cell": "c477_calib_negp_4",
            "seed": 42,
            "lr": 1e-5,
            "phase": "calibration",
            "assigned_gpu": 0,
            "status": "resumed_skip",
            # Missing source_self_delta_g_at_last_ckpt + source_emission_p_at_last_ckpt.
        }
    ]
    pick_path = tmp_path / "calibration_pick.json"
    with pytest.raises(KeyError):
        mod._phase_calibration_pick(bad_results, pick_path)


# ── partial_spearman defensive phase=="main" assert (cheap fix #3) ───────────


def test_partial_spearman_refuses_non_main_phase_cells():
    """Cheap fix #3: pooling implant-sweep cells into H1 partial MUST fail loud."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.analyze import (
        partial_spearman_count_given_implant,
    )

    # 6 main-phase cells (3 count levels × 2 seeds) PLUS 6 implant-sweep cells
    # at ANCHOR_COUNT=4. The implant-sweep cells would silently inject 6 extra
    # count=4 data points if not filtered — the defensive assert catches that.
    poisoned = []
    for seed in (42, 137):
        for cnt, by_dg in ((2, 4.0), (4, 5.0), (8, 6.0)):
            poisoned.append(
                {
                    "cell": f"c477_main_calib_negp_{cnt}",
                    "seed": seed,
                    "count": cnt,
                    "lr": 1e-5,
                    "phase": "main",
                    "source_self_delta_g_at_last_ckpt": 12.0 + 0.05 * cnt,
                    "source_emission_p_at_last_ckpt": 0.5,
                    "mean_bystander_delta_g": by_dg,
                    "step_at_last_ckpt": 100,
                }
            )
        # 3 implant-sweep cells at count=4 (the poison rows).
        for lr in (5e-6, 1e-5, 2e-5):
            poisoned.append(
                {
                    "cell": f"c477_implantsweep_lr{lr:g}",
                    "seed": seed,
                    "count": 4,
                    "lr": lr,
                    "phase": "implant_sweep",  # poison: NOT main
                    "source_self_delta_g_at_last_ckpt": 12.0,
                    "source_emission_p_at_last_ckpt": 0.5,
                    "mean_bystander_delta_g": 5.0,
                    "step_at_last_ckpt": 100,
                }
            )

    with pytest.raises(AssertionError, match="non-main-phase cell"):
        partial_spearman_count_given_implant(poisoned, n_bootstrap=50)
