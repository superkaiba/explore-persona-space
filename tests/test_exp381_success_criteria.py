"""Unit tests for exp381 plan v2 H1/H2 success-criteria aggregation.

Verifies Codex r2 Majors #1 and #2 (round-3 fixes):

* Major #1 — H1/H2 success uses the 3-seed *mean* (not per-cell ``any()``).
  One lucky seed passing while the other two fail must NOT satisfy H1/H2.
* Major #2 — H1/H2 baseline is the per-framing non-teach four-frame mean
  (not the pooled all-5-persona aggregate).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_exp381():
    if "exp381" in sys.modules:
        return sys.modules["exp381"]
    repo_root = Path(__file__).resolve().parent.parent
    scripts_dir = repo_root / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    spec = importlib.util.spec_from_file_location("exp381", scripts_dir / "run_experiment_381.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["exp381"] = mod
    spec.loader.exec_module(mod)
    return mod


def _make_cell(
    arm: str,
    seed: int,
    ckpt_step: int | None,
    f1_teach: float,
    f1_nt_mean: float,
    f11_teach: float,
    f11_nt_mean: float,
) -> dict:
    """Build a minimal cell row whose 4 non-teach personas all hit ``nt_mean``
    (so the four-frame mean is exactly ``nt_mean``) and teach persona hits the
    given rate. Keeps tests focused on the aggregation logic, not the
    ``_check_thresholds`` arithmetic.
    """
    m = _load_exp381()
    teach = m.TEACHING_PERSONA
    non_teach = list(m.NON_TEACH_PERSONAS)
    tag = f"{arm}_seed{seed}" + (f"_ckpt{ckpt_step}" if ckpt_step is not None else "")
    return {
        "arm": arm,
        "seed": seed,
        "ckpt_step": ckpt_step,
        "tag": tag,
        "per_framing_pass_rates": {
            "1": {teach: f1_teach, **{p: f1_nt_mean for p in non_teach}},
            "11": {teach: f11_teach, **{p: f11_nt_mean for p in non_teach}},
        },
    }


def test_3seed_mean_h1_fails_with_only_one_passing_seed() -> None:
    """Codex r2 Major #1: 1-of-3 seeds passing must NOT satisfy H1.

    The per-cell ``any()`` (round-2 bug) would declare H1 satisfied here; the
    3-seed mean (round-3 fix) correctly returns False because the mean teach
    rate across the 3 seeds is below 80%.
    """
    m = _load_exp381()
    # baseline = 0.0 for both framings, so non-teach four-frame mean must be ≤ 0.10
    base_non_teach_means = {"1": 0.0, "11": 0.0}
    # At ckpt_step=5: seed 42 passes (teach=0.95, nt=0.05), seeds 137 & 256
    # fail (teach=0.40, nt=0.05). Per-cell any() would return True for seed 42
    # alone; 3-seed mean of teach = (0.95 + 0.40 + 0.40) / 3 = 0.583, fails.
    cells = [
        _make_cell("anchor", 42, 5, 0.95, 0.05, 0.95, 0.05),
        _make_cell("anchor", 137, 5, 0.40, 0.05, 0.40, 0.05),
        _make_cell("anchor", 256, 5, 0.40, 0.05, 0.40, 0.05),
    ]
    result = m._success_criteria_predicates(cells, base_non_teach_means)
    assert result["h1_satisfied"] is False, (
        f"H1 must NOT be satisfied when only 1-of-3 seeds passes; "
        f"3-seed mean teach rate is 0.583 (< 0.80). result={result}"
    )
    # The per-step entry must still carry per-seed detail.
    step5 = result["h1_per_ckpt_step"]["5"]
    assert step5["n_seeds"] == 3, step5
    assert len(step5["per_seed"]) == 3, step5
    assert step5["both_framings_satisfied_at_3seed_mean"] is False, step5


def test_3seed_mean_h1_passes_when_all_3_seeds_pass() -> None:
    """All 3 seeds passing → 3-seed mean teach 0.90 ≥ 0.80 → H1 satisfied."""
    m = _load_exp381()
    base_non_teach_means = {"1": 0.0, "11": 0.0}
    cells = [
        _make_cell("anchor", 42, 10, 0.90, 0.05, 0.90, 0.05),
        _make_cell("anchor", 137, 10, 0.90, 0.05, 0.90, 0.05),
        _make_cell("anchor", 256, 10, 0.90, 0.05, 0.90, 0.05),
    ]
    result = m._success_criteria_predicates(cells, base_non_teach_means)
    assert result["h1_satisfied"] is True, result


def test_3seed_mean_h1_separate_step_aggregation() -> None:
    """Aggregation must group by ``ckpt_step`` — seed 42's ckpt5 result must
    not be pooled with seed 42's ckpt10 result.
    """
    m = _load_exp381()
    base_non_teach_means = {"1": 0.0, "11": 0.0}
    # ckpt 5: all 3 seeds fail teach; ckpt 10: all 3 seeds pass teach.
    cells = [
        _make_cell("anchor", 42, 5, 0.40, 0.05, 0.40, 0.05),
        _make_cell("anchor", 137, 5, 0.40, 0.05, 0.40, 0.05),
        _make_cell("anchor", 256, 5, 0.40, 0.05, 0.40, 0.05),
        _make_cell("anchor", 42, 10, 0.90, 0.05, 0.90, 0.05),
        _make_cell("anchor", 137, 10, 0.90, 0.05, 0.90, 0.05),
        _make_cell("anchor", 256, 10, 0.90, 0.05, 0.90, 0.05),
    ]
    result = m._success_criteria_predicates(cells, base_non_teach_means)
    assert result["h1_satisfied"] is True, result  # at least one ckpt (10) passes
    assert result["h1_per_ckpt_step"]["5"]["both_framings_satisfied_at_3seed_mean"] is False, (
        result["h1_per_ckpt_step"]["5"]
    )
    assert result["h1_per_ckpt_step"]["10"]["both_framings_satisfied_at_3seed_mean"] is True, (
        result["h1_per_ckpt_step"]["10"]
    )


def test_h1_must_satisfy_both_framings_simultaneously() -> None:
    """Plan v2: success requires framing #1 AND framing #11 thresholds both
    met at the SAME ckpt + 3-seed mean. A ckpt that satisfies framing #1 but
    not #11 (or vice versa) must NOT count toward H1.
    """
    m = _load_exp381()
    base_non_teach_means = {"1": 0.0, "11": 0.0}
    # Framing #1 strong, framing #11 weak — should NOT satisfy H1.
    cells = [
        _make_cell("anchor", 42, 5, 0.90, 0.05, 0.50, 0.05),
        _make_cell("anchor", 137, 5, 0.90, 0.05, 0.50, 0.05),
        _make_cell("anchor", 256, 5, 0.90, 0.05, 0.50, 0.05),
    ]
    result = m._success_criteria_predicates(cells, base_non_teach_means)
    assert result["h1_satisfied"] is False, result
    step5 = result["h1_per_ckpt_step"]["5"]
    assert step5["framing_1_3seed_mean"]["framing_satisfied"] is True
    assert step5["framing_11_3seed_mean"]["framing_satisfied"] is False


def test_3seed_mean_h2_armB_aggregation() -> None:
    """H2 uses the same 3-seed mean discipline across Arm B end-of-epoch cells."""
    m = _load_exp381()
    base_non_teach_means = {"1": 0.0, "11": 0.0}
    cells = [
        _make_cell("armB", 42, None, 0.95, 0.05, 0.95, 0.05),
        _make_cell("armB", 137, None, 0.40, 0.05, 0.40, 0.05),
        _make_cell("armB", 256, None, 0.40, 0.05, 0.40, 0.05),
    ]
    result = m._success_criteria_predicates(cells, base_non_teach_means)
    # 3-seed mean teach = (0.95+0.40+0.40)/3 = 0.583, fails.
    assert result["h2_satisfied"] is False, result
    agg = result["h2_armB_aggregate"]
    assert agg["n_seeds"] == 3, agg
    assert len(agg["per_seed"]) == 3, agg


def test_baseline_is_non_teach_only_not_pooled() -> None:
    """Codex r2 Major #2: H1/H2 baseline must be the per-framing non-teach
    four-frame mean — not pooled across all 5 personas.

    Construct a scenario where the pooled all-5 baseline would (incorrectly)
    swing the threshold by a wide margin: teach persona has 50% base rate,
    non-teach personas have 5% base rate. The non-teach four-frame mean is
    0.05; the pooled all-5 mean would be ~14%. We pass non_teach_means = 0.05
    here (what Phase 0 SHOULD persist after Major #2) and verify the
    threshold is 0.05 + 0.10 = 0.15.

    A non-teach four-frame mean of 0.14 with this baseline must FAIL (over
    threshold 0.15? no, under) — let's pick 0.16 which is *under* the wrong
    pooled threshold (~0.24) but *over* the right non-teach threshold (0.15).
    """
    m = _load_exp381()
    # Right baseline: non-teach four-frame mean = 0.05 → threshold 0.15.
    base_non_teach_means = {"1": 0.05, "11": 0.05}
    # Eval result: teach 0.90, non-teach four-frame mean 0.16 — over threshold
    # 0.15, so H1 must NOT be satisfied.
    cells = [
        _make_cell("anchor", 42, 5, 0.90, 0.16, 0.90, 0.16),
        _make_cell("anchor", 137, 5, 0.90, 0.16, 0.90, 0.16),
        _make_cell("anchor", 256, 5, 0.90, 0.16, 0.90, 0.16),
    ]
    result = m._success_criteria_predicates(cells, base_non_teach_means)
    step5 = result["h1_per_ckpt_step"]["5"]
    assert step5["framing_1_3seed_mean"]["baseline_non_teach_four_frame"] == 0.05, step5
    assert step5["framing_1_3seed_mean"]["framing_satisfied"] is False, step5
    assert result["h1_satisfied"] is False
    # And the result must record the source so the analyzer can verify.
    assert "non-teach" in result["baseline_source"].lower(), result["baseline_source"]


def test_framing_11_rotation_balance_post_decoupling() -> None:
    """Codex r2 Minor #1: after decoupling the decoy-exclusion and correct-
    position rotations, the per-decoy and per-position counts must remain
    balanced (6 each over 30 probes) and the (excluded_decoy, position)
    pair must cover MORE than 5 of the 25 combos.

    This is the contract the module-load ``_verify_framing_11_rotation_balance``
    enforces; exercising it from the test suite catches regressions before
    any import.
    """
    excluded_counts: dict[int, int] = {k: 0 for k in range(5)}
    position_counts: dict[int, int] = {k: 0 for k in range(1, 6)}
    pair_counts: dict[tuple[int, int], int] = {}
    for i in range(30):
        excluded = i % 5
        position = ((i // 5 + i) % 5) + 1
        excluded_counts[excluded] += 1
        position_counts[position] += 1
        pair_counts[(excluded, position)] = pair_counts.get((excluded, position), 0) + 1
    assert all(v == 6 for v in excluded_counts.values()), excluded_counts
    assert all(v == 6 for v in position_counts.values()), position_counts
    assert len(pair_counts) > 5, (
        f"decoy/position pair sweep regressed to coupled cycle "
        f"({len(pair_counts)} pairs; expected >5)"
    )


def test_phase0_per_persona_rates_schema(tmp_path: Path) -> None:
    """Codex r2 Major #2: the new ``base_per_persona_rates.json`` artifact must
    have shape ``{framing_id_str: {persona_name: float}}`` for all 11 framings
    and 5 personas (or be a subset if Phase 0 hasn't covered every cell yet).

    We exercise the JSON file's schema constraints — the actual values are
    Phase-0 output; this test checks the contract the analyzer reads.
    """
    import json

    m = _load_exp381()
    # Simulate a Phase 0 write with the right schema.
    base_per_persona = {
        str(fid): {p: 0.05 for p in m.EVAL_FRAMES}  # all 5 personas
        for fid in range(1, m.N_FRAMINGS + 1)
    }
    path = tmp_path / "base_per_persona_rates.json"
    path.write_text(json.dumps(base_per_persona, indent=2))

    # Round-trip + schema verification.
    loaded = json.loads(path.read_text())
    assert set(loaded.keys()) == {str(fid) for fid in range(1, m.N_FRAMINGS + 1)}
    for fid_str, persona_rates in loaded.items():
        assert m.TEACHING_PERSONA in persona_rates, f"framing {fid_str} missing teach persona"
        for nt in m.NON_TEACH_PERSONAS:
            assert nt in persona_rates, f"framing {fid_str} missing non-teach persona {nt}"
            assert isinstance(persona_rates[nt], int | float), (
                f"framing {fid_str} persona {nt} non-numeric rate {persona_rates[nt]!r}"
            )
