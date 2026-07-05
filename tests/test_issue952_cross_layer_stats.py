"""Unit pins for the issue #952 `cross-layer-decision-cells` registered statistics.

Pins the follow-up plan §2 constructions EXACTLY as registered: the pinned
bootstrap-p (shift-to-null-center, add-one counting; two-sided H1 / one-sided
positive-tail H2), Holm step-down, the three-way per-layer status enums
(NON-STRICT ±0.03 equivalence containment for H1; CI-upper < 0.02 affirmative
non-replication for H2), and the TOTAL outcome lattice — including the plan's
mechanizable check that `L20_local` can NEVER fire off indeterminate reads
(affirmative >= 2/3 in either family is the only route).

Also pins the fail-loud H2 decision-cell coverage contract (concern
`cross-layer-h2-missing-cells-silent-indeterminate`): a registered `M16_L*`
cell missing from the npz — including the producer's small-paired-universe
skip-branch shape (`M16_ctx_ids` written, ZERO per-layer M arrays) — makes
production `--cross-layer` RAISE before any statistic; `stats_cross_layer.json`
is NEVER written with a `read_missing` record.
"""

import argparse
import importlib.util
import itertools
import json
import pathlib

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "issue952_stats_under_test",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "issue952_stats.py",
)
stats = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(stats)

STATUSES = ("replicates", "affirmative_nonreplication", "indeterminate")


def test_pinned_p_add_one_counting_two_sided():
    """p = (1 + #{|null| >= |obs|}) / (1 + B) with null = draws - observed."""
    obs = 1.0
    # draws - obs = [-2, -1, 0, 1, 2] -> |null| >= 1 for 4 of 5 draws.
    draws = np.asarray([-1.0, 0.0, 1.0, 2.0, 3.0])
    p = stats.pinned_bootstrap_p(obs, draws, tail="two")
    assert p == pytest.approx((1 + 4) / (1 + 5))


def test_pinned_p_one_sided_positive_tail():
    """H2 tail: p = (1 + #{null >= obs}) / (1 + B)."""
    obs = 1.0
    # null = [-2, -1, 0, 1, 2] -> null >= 1 for 2 of 5 draws.
    draws = np.asarray([-1.0, 0.0, 1.0, 2.0, 3.0])
    p = stats.pinned_bootstrap_p(obs, draws, tail="greater")
    assert p == pytest.approx((1 + 2) / (1 + 5))


def test_pinned_p_guards():
    """Non-finite observed / empty draws -> None (dropped, never coerced)."""
    assert stats.pinned_bootstrap_p(float("nan"), np.asarray([1.0]), tail="two") is None
    assert stats.pinned_bootstrap_p(1.0, np.asarray([]), tail="two") is None
    with pytest.raises(ValueError):
        stats.pinned_bootstrap_p(1.0, np.asarray([1.0]), tail="less")


def test_holm_step_down_matches_registered_convention():
    """Holm: sort ascending; adj = max-running (k - rank) * p, capped at 1."""
    out = stats.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"] == pytest.approx(0.03)  # 3 * 0.01
    assert out["c"] == pytest.approx(0.06)  # max(0.03, 2 * 0.03)
    assert out["b"] == pytest.approx(0.06)  # max(0.06, 1 * 0.04)
    assert stats.holm_adjust({"a": 0.9, "b": 0.8})["a"] == 1.0  # cap
    assert stats.holm_adjust({"a": 0.01, "b": None})["b"] is None  # missing read


def test_h1_status_nonstrict_containment_boundary():
    """Equivalence containment is NON-STRICT: lo == -0.03 AND hi == +0.03 replicates."""
    assert stats.h1_status(0.0, -0.03, 0.03, holm_p=1.0) == "replicates"
    # Straddling a margin boundary without a Holm-significant detection.
    assert stats.h1_status(0.02, -0.05, 0.09, holm_p=0.5) == "indeterminate"
    # Affirmative route 1: contrast >= +0.03 AND Holm p < 0.05.
    assert stats.h1_status(0.05, 0.01, 0.09, holm_p=0.01) == "affirmative_nonreplication"
    # Same point estimate WITHOUT the Holm detection -> indeterminate.
    assert stats.h1_status(0.05, 0.01, 0.09, holm_p=0.20) == "indeterminate"
    # Affirmative route 2: CI entirely outside the band (either side).
    assert stats.h1_status(0.06, 0.035, 0.09, holm_p=1.0) == "affirmative_nonreplication"
    assert stats.h1_status(-0.06, -0.09, -0.035, holm_p=1.0) == "affirmative_nonreplication"
    # Missing read -> indeterminate.
    assert stats.h1_status(None, float("nan"), float("nan"), None) == "indeterminate"


def test_h2_status_enum():
    """Replicates = ΔG >= 0.02 AND Holm p < 0.05; affirmative = CI upper < 0.02."""
    assert stats.h2_status(0.03, hi=0.05, holm_p=0.01) == "replicates"
    assert stats.h2_status(0.03, hi=0.05, holm_p=0.20) == "indeterminate"
    assert stats.h2_status(0.005, hi=0.015, holm_p=0.5) == "affirmative_nonreplication"
    assert stats.h2_status(0.01, hi=0.05, holm_p=0.5) == "indeterminate"
    assert stats.h2_status(None, hi=float("nan"), holm_p=None) == "indeterminate"


def test_lattice_totality_and_forbidden_route():
    """Every 3^3 x 3^3 status combination maps to exactly one of the four
    registered outcomes, and `L20_local` requires >= 2/3 affirmative in a family
    (indeterminate reads NEVER open it) — plan §2 mechanizable check."""
    layers = ("14", "23", "26")
    verdicts = set()
    for combo1 in itertools.product(STATUSES, repeat=3):
        for combo2 in itertools.product(STATUSES, repeat=3):
            h1 = dict(zip(layers, combo1, strict=True))
            h2 = dict(zip(layers, combo2, strict=True))
            rec = stats.map_outcome_lattice(h1, h2)
            v = rec["overall_verdict"]
            verdicts.add(v)
            assert v in {
                "full_replication",
                "L20_local",
                "inconclusive_layer_scope",
                "partial_band_map",
            }
            aff1 = combo1.count("affirmative_nonreplication")
            aff2 = combo2.count("affirmative_nonreplication")
            if v == "L20_local":
                assert max(aff1, aff2) >= 2  # the ONLY route
            if max(aff1, aff2) >= 2:
                assert v == "L20_local"
            if v == "full_replication":
                assert combo1 == ("replicates",) * 3 and combo2 == ("replicates",) * 3
            if v == "inconclusive_layer_scope":
                assert combo1 == ("indeterminate",) * 3 and combo2 == ("indeterminate",) * 3
    assert verdicts == {
        "full_replication",
        "L20_local",
        "inconclusive_layer_scope",
        "partial_band_map",
    }


def test_lattice_rejects_mismatched_layer_sets():
    """The lattice needs the SAME non-empty added-layer set in both families."""
    with pytest.raises(AssertionError):
        stats.map_outcome_lattice({"14": "replicates"}, {"23": "replicates"})
    with pytest.raises(AssertionError):
        stats.map_outcome_lattice({}, {})


# ── fail-loud H2 decision-cell coverage (concern
#    cross-layer-h2-missing-cells-silent-indeterminate) ──────────────────────────

XLAYER_LAYERS = (14, 20, 23, 26)  # family {14, 23, 26} + production cal layer 20
_TEST_IDS = list(range(6))


def _minimal_xlayer_npz() -> dict[str, np.ndarray]:
    """Minimal COMPLETE cross-layer npz: suffixed A blocks (real F16/L16 slot x
    arm group names, so the H1 gap cells resolve) + the full registered M16
    decision-cell grid over XLAYER_LAYERS. Bank arrays deliberately absent
    (H3 degrades to its documented `bank_arrays_absent` descriptive record)."""
    rng = np.random.default_rng(7)
    slots = list(stats.F16_SLOTS) + list(stats.L16_CONTENT_SLOTS)
    groups = [f"{s}|{a}" for s in slots for a in ("own", "ext_plain", "ext_style")]
    n = len(_TEST_IDS)
    npz: dict[str, np.ndarray] = {
        "A_test_ctx_ids": np.asarray(_TEST_IDS, dtype=np.int64),
        "A_group_names": np.asarray(groups),
        "M16_ctx_ids": np.asarray(_TEST_IDS, dtype=np.int64),
    }
    for la in XLAYER_LAYERS:
        npz[f"A_test_ssres_L{la}"] = rng.uniform(0.1, 0.9, size=(n, len(groups)))
        npz[f"A_test_sstot_L{la}"] = rng.uniform(0.5, 1.5, size=(n, len(groups)))
        for leg in ("cleg", "zleg"):
            for arm in ("own", "ext_plain", "ext_style"):
                npz[f"M16_L{la}_{leg}_{arm}_ssres"] = rng.uniform(0.1, 0.9, size=n)
                npz[f"M16_L{la}_{leg}_{arm}_sstot"] = rng.uniform(0.5, 1.5, size=n)
    return npz


def _write_xlayer_fixture(tmp_path: pathlib.Path, npz: dict[str, np.ndarray]) -> argparse.Namespace:
    """Stage the minimal eval-dir tree + spans + parent stats; return the
    PRODUCTION-mode (smoke=False) argparse namespace for cross_layer_main."""
    eval_dir = tmp_path / "eval"
    eval_dir.mkdir()
    (eval_dir / "battery_meta.json").write_text(json.dumps({"l_star_pos": 20}))
    (eval_dir / "split_seed952.json").write_text(
        json.dumps({"train": [], "val": [], "test": _TEST_IDS})
    )
    (eval_dir / "divergence_bank_verification.json").write_text(
        json.dumps({"kept_pairs": [], "pairs": []})
    )
    spans_dir = tmp_path / "spans"
    spans_dir.mkdir()
    for arm in stats.ARMS:
        (spans_dir / f"spans_{arm}.json").write_text(
            json.dumps({str(c): {"span": 200} for c in _TEST_IDS})
        )
    parent_stats = tmp_path / "parent_stats_summary.json"
    parent_stats.write_text("{}")
    npz_path = tmp_path / "per_context_stats_cross_layer.npz"
    np.savez(npz_path, **npz)
    return argparse.Namespace(
        eval_dir=str(eval_dir),
        npz=str(npz_path),
        spans_dir=str(spans_dir),
        out=str(tmp_path / "stats_cross_layer.json"),
        n_draws=200,
        smoke=False,
        decision_layers="14,23,26",
        parent_stats=str(parent_stats),
    )


def test_missing_h2_cell_raises_in_production(tmp_path):
    """REGRESSION (reconciler round-5 BLOCKER): delete ONE required registered
    key -> production --cross-layer RAISES at the coverage gate; the stats JSON
    is never written (no silent `read_missing`/`indeterminate` downgrade)."""
    npz = _minimal_xlayer_npz()
    del npz["M16_L23_zleg_ext_plain_ssres"]
    args = _write_xlayer_fixture(tmp_path, npz)
    with pytest.raises(RuntimeError, match="H2 decision-cell coverage FAIL"):
        stats.cross_layer_main(args)
    assert not pathlib.Path(args.out).exists()


def test_producer_skip_branch_shape_raises_in_production(tmp_path):
    """The reconciler-traced reachable shape: the producer's small-paired-universe
    skip branch writes `M16_ctx_ids` while skipping EVERY per-layer M array —
    the enumeration must catch exactly that npz (it passes both legacy coverage
    asserts) and raise before any statistic."""
    npz = {
        k: v
        for k, v in _minimal_xlayer_npz().items()
        if not (k.startswith("M16_L"))  # keep M16_ctx_ids, drop all per-layer M cells
    }
    args = _write_xlayer_fixture(tmp_path, npz)
    with pytest.raises(RuntimeError, match="H2 decision-cell coverage FAIL"):
        stats.cross_layer_main(args)
    assert not pathlib.Path(args.out).exists()


def test_cross_layer_main_completes_on_full_grid(tmp_path):
    """Companion PASS pin: on the COMPLETE registered cell grid the production
    driver runs end-to-end — every added layer carries a finite raw + Holm p
    and a status, and NO `read_missing` record exists anywhere in the JSON."""
    args = _write_xlayer_fixture(tmp_path, _minimal_xlayer_npz())
    stats.cross_layer_main(args)
    out = json.loads(pathlib.Path(args.out).read_text())
    assert "read_missing" not in json.dumps(out)
    assert sorted(out["h2_by_layer"]) == ["14", "23", "26"]
    for la in ("14", "23", "26"):
        rec = out["h2_by_layer"][la]["ext_plain"]
        assert np.isfinite(rec["p_one_sided_raw"]) and np.isfinite(rec["p_holm"])
        assert rec["status"] in STATUSES
        assert np.isfinite(out["h1_by_layer"][la]["ext_plain"]["p_two_sided_raw"])
    assert out["row_coverage"]["h2_decision_cells"]["layers"] == [14, 20, 23, 26]
    assert out["overall_verdict"] in {
        "full_replication",
        "L20_local",
        "inconclusive_layer_scope",
        "partial_band_map",
    }


def test_h2_layer_read_raises_on_missing_cell():
    """Defense-in-depth behind the coverage gate: a bank without the registered
    M cells makes h2_layer_read RAISE — it never returns a partial/None read."""
    bank = stats.CellBank([0, 1])
    bank.add("A_L14|f16_t1|own", np.asarray([0, 1]), np.asarray([0.5, 0.4]), np.asarray([1.0, 1.0]))
    obs = bank.observed()
    draws = bank.draws(np.ones((3, 2)))
    with pytest.raises(RuntimeError, match="M\\|16_L14_cleg_own"):
        stats.h2_layer_read(bank, obs, draws, layer=14)


def test_registered_family_ps_gate():
    """A None / non-finite raw p in a registered family must fail loud BEFORE
    holm_adjust (never silently shrink k_tests)."""
    stats._assert_registered_family_ps("h2", {"14": 0.02, "23": 0.5, "26": 0.9})  # OK
    with pytest.raises(RuntimeError, match="h2 registered family incomplete"):
        stats._assert_registered_family_ps("h2", {"14": 0.02, "23": None, "26": 0.9})
    with pytest.raises(RuntimeError, match="h1 registered family incomplete"):
        stats._assert_registered_family_ps("h1", {"14": float("nan"), "23": 0.5, "26": 0.9})
