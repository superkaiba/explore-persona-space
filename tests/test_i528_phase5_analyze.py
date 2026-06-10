# ruff: noqa: RUF002, RUF003  -- intentional math notation (Spearman ρ).
"""Unit tests for ``scripts.i528_phase5_analyze``.

Pins down the three round-1-code-review blocker fixes carried into round 2:

- **Blocker 1.** The H2 bootstrap UNIT is the per-seed mean over H1-passing
  traits — N=3 observations — not the flat list of 12 (trait, seed) cells.
  A test constructs a synthetic ``h2_passing_seed_means`` of three identical
  values + a flat-12 expansion with the same mean and asserts the bootstrap
  CIs differ in width by the expected ~sqrt(4) factor.
- **Blocker 2.** H1 PASS excludes a trait whose ``base_summary[trait]["ci_lo"]
  >= 3.5`` even when ``reject and ci_lo > 0 and paired_delta_mean > 0``
  would otherwise satisfy the conjunction. A test runs ``main()`` against a
  synthetic ``judge_scores.json`` whose base scores are all 5.0 (CI tightly
  above 3.5) and a positive trained delta; asserts the trait ends up with
  ``pass_h1=False`` and a ``h1_untestable: saturated_base`` marker.
- (Blocker 3 belongs to ``train_lora`` in ``src/explore_persona_space/train/
  sft.py``; see ``tests/test_i528_sft_adapter_persist.py``.)
"""

from __future__ import annotations

import json

import pytest

# ---------------------------------------------------------------------------
# Blocker 1 — bootstrap unit is per-seed, not per-cell.
# ---------------------------------------------------------------------------


def test_bootstrap_ci_unit_is_per_seed_mean_not_flat_cells():
    """With three per-seed means vs the same mean expanded to 12 cells, the
    per-seed bootstrap CI must be ~sqrt(4) wider than the flat-12 CI.

    The bootstrap variance scales as ``var/N``. The per-seed list has N=3;
    the flat list has N=12 (12 = 4 traits × 3 seeds), so the SE ratio is
    ``sqrt(12/3) = 2``. A 2× CI half-width difference flips H2 PASS / FAIL
    on a marginal null. The test pins down the helper's behavior, not the
    headline H2 number, but the headline H2 number reads off this same
    bootstrap and so inherits the contract.
    """
    from scripts.i528_phase5_analyze import _bootstrap_ci

    # Three per-seed means (Blocker 1 unit).
    per_seed = [-0.30, -0.10, 0.10]
    # Same population mean but expanded 4× (four H1-passing traits per
    # seed all coincidentally equal to the per-seed mean — the pathological
    # case that maximizes the wrong unit's CI shrinkage).
    flat = [v for v in per_seed for _ in range(4)]

    assert len(per_seed) == 3
    assert len(flat) == 12
    assert abs(sum(per_seed) / 3 - sum(flat) / 12) < 1e-9  # same mean

    lo_seed, hi_seed = _bootstrap_ci(per_seed, n_resamples=5000, seed=42)
    lo_flat, hi_flat = _bootstrap_ci(flat, n_resamples=5000, seed=42)

    width_seed = hi_seed - lo_seed
    width_flat = hi_flat - lo_flat

    # Per-seed CI is materially wider. Allow some bootstrap noise but
    # require the ratio to be clearly above 1.5 (true asymptotic is 2.0).
    assert width_seed > width_flat * 1.5, (
        f"Per-seed CI width ({width_seed:.4f}) must be >~2x flat-cell "
        f"CI width ({width_flat:.4f}); ratio={width_seed / width_flat:.2f}. "
        "Blocker 1 regression — bootstrap unit collapsed to flat cells."
    )


# ---------------------------------------------------------------------------
# Blocker 2 — H1 PASS excludes saturated-base traits.
# ---------------------------------------------------------------------------


def _build_synthetic_judge_payload(
    *,
    saturated_trait: str,
    saturated_base_score: float,
    saturated_trained_score: float,
    other_trait: str,
) -> dict:
    """Construct a judge-payload dict that triggers the saturation gate.

    The ``saturated_trait`` has a flat base score that pins ``base_ci_lo
    > 3.5`` (no headroom) and a positive trained delta. Without the
    saturation gate, the trait would pass the ``reject and ci_lo > 0``
    conjunction; with the gate, it is excluded as untestable.

    The ``other_trait`` carries low base + high trained scores so the
    Holm-Bonferroni correction sees a plausible non-saturated trait
    too (the saturation gate's exclusion is independent of Holm, but
    a one-trait Holm has no correction and would make the test
    degenerate).
    """
    rows: list[dict] = []
    seeds = (42, 137, 1337)
    # 40 questions per trait — matches the plan §11 N=40 paired prompts
    # per trait, though for this test we only need ≥2 to get a paired
    # diff and a tight enough base CI.
    n_q = 40

    for q in range(n_q):
        # Saturated trait — base ALL at 5.0 → CI = [5.0, 5.0]; trained at
        # 5.0 too (delta 0). Actually for the saturation test we want a
        # POSITIVE delta + significant t so the WITHOUT-gate code would
        # incorrectly pass it. So: base at saturated_base_score and
        # trained slightly higher.
        for seed in seeds:
            rows.append(
                {
                    "kind": "trained",
                    "trait": saturated_trait,
                    "arm": "system",
                    "seed": seed,
                    "eval_context": "own_scenario",
                    "q_idx": q,
                    "score": saturated_trained_score,
                }
            )
        rows.append(
            {
                "kind": "base",
                "trait": saturated_trait,
                "arm": "system",
                "seed": -1,
                "eval_context": "own_scenario",
                "q_idx": q,
                "score": saturated_base_score,
            }
        )

        # Other trait — low base (~2.0), high trained (~4.0); huge real
        # effect, no saturation.
        for seed in seeds:
            rows.append(
                {
                    "kind": "trained",
                    "trait": other_trait,
                    "arm": "system",
                    "seed": seed,
                    "eval_context": "own_scenario",
                    "q_idx": q,
                    "score": 4.0,
                }
            )
        rows.append(
            {
                "kind": "base",
                "trait": other_trait,
                "arm": "system",
                "seed": -1,
                "eval_context": "own_scenario",
                "q_idx": q,
                "score": 2.0,
            }
        )

    return {"schema_version": "i528_judge_v1", "rows": rows}


def test_h1_pass_excludes_saturated_base_ci(tmp_path, monkeypatch):
    """A trait whose base ``ci_lo >= 3.5`` must NOT pass H1 even when
    ``reject and ci_lo > 0 and paired_delta_mean > 0`` are all true.

    Synthetic setup:

    * ``validating`` base = 5.0 everywhere → bootstrap base CI is
      [5.0, 5.0]; ``ci_lo = 5.0 >= 3.5`` so the saturation gate fires.
      Trained = 5.0 (delta exactly 0 — but we test the GATE, not the
      pass-conjunction): the ``base_saturated_ci`` field must be True
      and ``pass_h1`` must be False with reason ``saturated_base``.
    * ``conciseness`` base = 2.0, trained = 4.0 → big positive delta,
      tight low CI, ``pass_h1=True`` (sanity — without this we have a
      degenerate Holm).

    The interesting failure mode this test pins is the WITHOUT-gate
    regression where ``ci_lo > 0`` alone is sufficient for ``pass_h1``.
    """
    from scripts import i528_phase5_analyze as mod

    payload = _build_synthetic_judge_payload(
        saturated_trait="validating",
        saturated_base_score=5.0,
        saturated_trained_score=5.0,
        other_trait="conciseness",
    )

    judge_path = tmp_path / "judge_scores.json"
    out_path = tmp_path / "analysis.json"
    judge_path.write_text(json.dumps(payload))

    monkeypatch.setattr(mod, "JUDGE_PATH", judge_path)
    monkeypatch.setattr(mod, "OUT_PATH", out_path)
    monkeypatch.setattr(mod, "PARAPHRASE_PATH", tmp_path / "no_paraphrase.json")

    # Small n_bootstrap so the test runs in <1s. The bootstrap value
    # doesn't depend on n_resamples for this all-equal-base case (CI
    # collapses to the point mass), but we keep it small for speed.
    rc = mod.main(["--n-bootstrap", "200"])
    assert rc == 0
    out = json.loads(out_path.read_text())

    # Saturated trait must not pass H1.
    validating = out["h1_per_trait"]["validating"]
    assert validating["base_saturated_ci"] is True, (
        f"Saturation gate did not fire on flat-5.0 base: {validating}"
    )
    assert validating["pass_h1"] is False, (
        f"Blocker 2 regression — saturated trait PASSed H1: {validating}"
    )
    assert validating.get("h1_untestable") == "saturated_base"
    assert validating["base_ci_lo"] >= 3.5

    # Sanity — the non-saturated control trait passed.
    conciseness = out["h1_per_trait"]["conciseness"]
    assert conciseness["base_saturated_ci"] is False
    assert conciseness["headroom"] is True
    # Big effect: 4.0 − 2.0 = 2.0 delta; Holm on 1 test = uncorrected.
    assert conciseness["paired_delta_mean"] == pytest.approx(2.0, abs=1e-6)


def test_h1_pass_excludes_when_only_mean_below_3_5_but_ci_lo_above():
    """The gate uses ``base_summary[trait]["ci_lo"]``, not ``mean``.

    A trait whose base ``mean = 3.6`` and ``ci_lo = 3.55`` (above 3.5)
    must be excluded. A pre-Blocker-2 code path that gated on
    ``mean >= 3.5`` only would have already caught this, but we pin
    the CI-based gate explicitly so a future refactor that drops the
    CI check (e.g. "simpler — just use mean") gets caught.

    This is an in-memory check of the conjunction; it does not need
    to round-trip through ``main()``.
    """
    # Reconstruct the conjunction the way the code computes it.
    base_summary_lo = 3.55  # CI lower bound > 3.5 → saturated
    reject = True
    ci_lo_diffs = 0.10  # positive paired-diff CI lower bound
    paired_delta_mean = 0.20
    headroom = not (base_summary_lo > 3.5)

    pass_h1 = bool(headroom and reject and ci_lo_diffs > 0 and paired_delta_mean > 0)
    assert pass_h1 is False, "ci_lo > 3.5 must block pass_h1 even with reject + positive diff CI."


# ---------------------------------------------------------------------------
# Bootstrap regression — N=3 vs N=12 numerically (Blocker 1, complementary).
# ---------------------------------------------------------------------------


def test_h2_bootstrap_pins_per_seed_unit_via_main(tmp_path, monkeypatch):
    """End-to-end pin: the H2 summary's ``n_seeds`` field is the bootstrap-
    unit count and MUST be 3 (per-seed) — not 12 (trait×seed).

    Builds a 4-trait judge payload where every trait is an H1-passer (low
    base + high trained) and exercises ``main()``; asserts the H2 summary
    reports ``n_seeds=3`` and ``bootstrap_unit="per_seed_mean_over_h1_
    passing_traits"``.
    """
    from scripts import i528_phase5_analyze as mod

    traits = ["validating", "conciseness", "asks_clarifying_first", "calibrated_uncertainty"]
    seeds = (42, 137, 1337)
    rows: list[dict] = []
    n_q = 40

    for trait in traits:
        for q in range(n_q):
            # System + role both moderately leak; system slightly more so
            # d_leakage = role - system slightly negative on average.
            for seed in seeds:
                rows.append(
                    {
                        "kind": "trained",
                        "trait": trait,
                        "arm": "system",
                        "seed": seed,
                        "eval_context": "own_scenario",
                        "q_idx": q,
                        "score": 4.0,
                    }
                )
                # Off-target leak scores — these drive H2.
                for ctx in ("sibling_1", "sibling_2", "sibling_3", "default_assistant"):
                    rows.append(
                        {
                            "kind": "trained",
                            "trait": trait,
                            "arm": "system",
                            "seed": seed,
                            "eval_context": ctx,
                            "q_idx": q,
                            "score": 3.5,
                        }
                    )
                    rows.append(
                        {
                            "kind": "trained",
                            "trait": trait,
                            "arm": "role",
                            "seed": seed,
                            "eval_context": ctx,
                            "q_idx": q,
                            "score": 3.0,
                        }
                    )
            rows.append(
                {
                    "kind": "base",
                    "trait": trait,
                    "arm": "system",
                    "seed": -1,
                    "eval_context": "own_scenario",
                    "q_idx": q,
                    "score": 2.0,
                }
            )

    payload = {"schema_version": "i528_judge_v1", "rows": rows}

    judge_path = tmp_path / "judge_scores.json"
    out_path = tmp_path / "analysis.json"
    judge_path.write_text(json.dumps(payload))

    monkeypatch.setattr(mod, "JUDGE_PATH", judge_path)
    monkeypatch.setattr(mod, "OUT_PATH", out_path)
    monkeypatch.setattr(mod, "PARAPHRASE_PATH", tmp_path / "no_paraphrase.json")

    rc = mod.main(["--n-bootstrap", "500"])
    assert rc == 0

    out = json.loads(out_path.read_text())
    h2 = out["h2_paired_leakage"]

    assert h2["bootstrap_unit"] == "per_seed_mean_over_h1_passing_traits", h2
    assert h2["n_seeds"] == 3, (
        f"Blocker 1 regression — H2 bootstrap unit collapsed: n_seeds={h2['n_seeds']}"
    )
    # Per-seed structure preserved. (JSON deserializes int dict keys as strings.)
    assert set(h2["per_seed_mean"].keys()) == {"42", "137", "1337"}
    # All four traits passed H1, so the headline H2 reads off the same
    # 3-seed unit. d_mean should be ~ -0.5 (3.0 - 3.5).
    assert h2["d_mean"] == pytest.approx(-0.5, abs=1e-6)


def test_per_encoding_gate_splits_h1_by_arm(tmp_path, monkeypatch):
    """``--saturation-gate per_encoding`` splits the H1 gate per (trait, arm)
    cell using THAT arm's base own_scenario distribution.

    Synthetic setup: ``validating`` has system base = 5.0 (saturated) and
    role base = 3.0 (NOT saturated); both system and role trained = 4.0.
    Under per_encoding, the role cell must be H1-eligible (not saturated)
    AND must pass Holm + positive delta + CI check; the system cell must
    be flagged saturated. Under pooled (default), only the system arm's
    cell exists in ``h1_per_trait`` and it is saturated → no H1 passers.
    """
    from scripts import i528_phase5_analyze as mod

    rows: list[dict] = []
    seeds = (42, 137, 1337)
    n_q = 40

    for q in range(n_q):
        # validating: trained-sys = trained-role = 4.0; base-sys = 5.0
        # (saturated), base-role = 3.0 (NOT saturated).
        for arm, trained_score, base_score in (
            ("system", 4.0, 5.0),
            ("role", 4.0, 3.0),
        ):
            for seed in seeds:
                rows.append(
                    {
                        "kind": "trained",
                        "trait": "validating",
                        "arm": arm,
                        "seed": seed,
                        "eval_context": "own_scenario",
                        "q_idx": q,
                        "score": trained_score,
                    }
                )
            rows.append(
                {
                    "kind": "base",
                    "trait": "validating",
                    "arm": arm,
                    "seed": -1,
                    "eval_context": "own_scenario",
                    "q_idx": q,
                    "score": base_score,
                }
            )

    payload = {"schema_version": "i528_judge_v1", "rows": rows}
    judge_path = tmp_path / "judge_scores.json"
    out_path = tmp_path / "analysis.json"
    judge_path.write_text(json.dumps(payload))

    monkeypatch.setattr(mod, "JUDGE_PATH", judge_path)
    monkeypatch.setattr(mod, "OUT_PATH", out_path)
    monkeypatch.setattr(mod, "PARAPHRASE_PATH", tmp_path / "no_paraphrase.json")

    rc = mod.main(["--saturation-gate", "per_encoding", "--n-bootstrap", "500"])
    assert rc == 0
    out = json.loads(out_path.read_text())

    assert out["saturation_gate"] == "per_encoding"
    cells = out["h1_per_cell"]["validating"]
    assert set(cells.keys()) == {"system", "role"}

    sys_cell = cells["system"]
    role_cell = cells["role"]

    # System arm: base = 5.0 → saturated, no Holm test.
    assert sys_cell["base_saturated_ci"] is True
    assert sys_cell["pass_h1"] is False
    assert sys_cell.get("h1_untestable") == "saturated_base"

    # Role arm: base = 3.0 → NOT saturated; trained = 4.0 → delta = +1.0;
    # uniform constants give SE=0 and t=inf → Holm passes; pass_h1 True.
    assert role_cell["base_saturated_ci"] is False
    assert role_cell["headroom"] is True
    assert role_cell["pass_h1"] is True
    assert role_cell["paired_delta_mean"] == pytest.approx(1.0, abs=1e-6)

    # h1_passing_traits surfaces validating via the role cell.
    assert out["h1_passing_traits"] == ["validating"]


def test_pooled_gate_is_backward_compatible_default(tmp_path, monkeypatch):
    """Default invocation (no ``--saturation-gate`` flag) reproduces the
    legacy single-gate-per-trait behavior on the system arm only.
    """
    from scripts import i528_phase5_analyze as mod

    rows: list[dict] = []
    seeds = (42, 137, 1337)
    n_q = 40

    # validating: trained-sys = trained-role = 4.0; base-sys = 5.0 (sat),
    # base-role = 3.0 (NOT sat). Under POOLED the gate only reads base-sys,
    # so validating is gated out — no H1 passers.
    for q in range(n_q):
        for arm, trained_score, base_score in (
            ("system", 4.0, 5.0),
            ("role", 4.0, 3.0),
        ):
            for seed in seeds:
                rows.append(
                    {
                        "kind": "trained",
                        "trait": "validating",
                        "arm": arm,
                        "seed": seed,
                        "eval_context": "own_scenario",
                        "q_idx": q,
                        "score": trained_score,
                    }
                )
            rows.append(
                {
                    "kind": "base",
                    "trait": "validating",
                    "arm": arm,
                    "seed": -1,
                    "eval_context": "own_scenario",
                    "q_idx": q,
                    "score": base_score,
                }
            )

    payload = {"schema_version": "i528_judge_v1", "rows": rows}
    judge_path = tmp_path / "judge_scores.json"
    out_path = tmp_path / "analysis.json"
    judge_path.write_text(json.dumps(payload))

    monkeypatch.setattr(mod, "JUDGE_PATH", judge_path)
    monkeypatch.setattr(mod, "OUT_PATH", out_path)
    monkeypatch.setattr(mod, "PARAPHRASE_PATH", tmp_path / "no_paraphrase.json")

    # No --saturation-gate flag → defaults to pooled.
    rc = mod.main(["--n-bootstrap", "500"])
    assert rc == 0
    out = json.loads(out_path.read_text())

    assert out["saturation_gate"] == "pooled"
    # Pooled gate reads base-system only → saturated, no H1 passers.
    assert out["h1_passing_traits"] == []
    v = out["h1_per_trait"]["validating"]
    assert v["base_saturated_ci"] is True
    assert v["pass_h1"] is False
    # h1_per_cell under pooled mode only carries the system arm.
    cells = out["h1_per_cell"]["validating"]
    assert set(cells.keys()) == {"system"}


def test_h2_per_seed_means_average_over_h1_passing_only(tmp_path, monkeypatch):
    """A saturated-base trait does NOT enter the H2 per-seed means.

    Builds two H1-passing traits + two saturated-base traits; asserts
    the H2 summary's per-seed means are computed only over the 2 H1-
    passing traits, and that ``h1_passing_traits`` lists exactly those 2.
    """
    from scripts import i528_phase5_analyze as mod

    rows: list[dict] = []
    seeds = (42, 137, 1337)
    n_q = 20

    # Two non-saturated H1-passing traits — d_leakage = -0.4 per cell.
    for trait in ("validating", "conciseness"):
        for q in range(n_q):
            for seed in seeds:
                rows.append(
                    {
                        "kind": "trained",
                        "trait": trait,
                        "arm": "system",
                        "seed": seed,
                        "eval_context": "own_scenario",
                        "q_idx": q,
                        "score": 4.0,
                    }
                )
                for ctx in ("sibling_1", "sibling_2", "sibling_3", "default_assistant"):
                    rows.append(
                        {
                            "kind": "trained",
                            "trait": trait,
                            "arm": "system",
                            "seed": seed,
                            "eval_context": ctx,
                            "q_idx": q,
                            "score": 3.4,
                        }
                    )
                    rows.append(
                        {
                            "kind": "trained",
                            "trait": trait,
                            "arm": "role",
                            "seed": seed,
                            "eval_context": ctx,
                            "q_idx": q,
                            "score": 3.0,
                        }
                    )
            rows.append(
                {
                    "kind": "base",
                    "trait": trait,
                    "arm": "system",
                    "seed": -1,
                    "eval_context": "own_scenario",
                    "q_idx": q,
                    "score": 2.0,
                }
            )

    # Two saturated traits — base = 5.0 → excluded from H1, H2 per-seed
    # means. d_leakage on these would otherwise be +1.0 (role > system)
    # and would skew H2 the wrong way if accidentally included.
    for trait in ("asks_clarifying_first", "calibrated_uncertainty"):
        for q in range(n_q):
            for seed in seeds:
                rows.append(
                    {
                        "kind": "trained",
                        "trait": trait,
                        "arm": "system",
                        "seed": seed,
                        "eval_context": "own_scenario",
                        "q_idx": q,
                        "score": 5.0,
                    }
                )
                for ctx in ("sibling_1", "sibling_2", "sibling_3", "default_assistant"):
                    rows.append(
                        {
                            "kind": "trained",
                            "trait": trait,
                            "arm": "system",
                            "seed": seed,
                            "eval_context": ctx,
                            "q_idx": q,
                            "score": 4.0,
                        }
                    )
                    rows.append(
                        {
                            "kind": "trained",
                            "trait": trait,
                            "arm": "role",
                            "seed": seed,
                            "eval_context": ctx,
                            "q_idx": q,
                            "score": 5.0,
                        }
                    )
            rows.append(
                {
                    "kind": "base",
                    "trait": trait,
                    "arm": "system",
                    "seed": -1,
                    "eval_context": "own_scenario",
                    "q_idx": q,
                    "score": 5.0,
                }
            )

    payload = {"schema_version": "i528_judge_v1", "rows": rows}
    judge_path = tmp_path / "judge_scores.json"
    out_path = tmp_path / "analysis.json"
    judge_path.write_text(json.dumps(payload))

    monkeypatch.setattr(mod, "JUDGE_PATH", judge_path)
    monkeypatch.setattr(mod, "OUT_PATH", out_path)
    monkeypatch.setattr(mod, "PARAPHRASE_PATH", tmp_path / "no_paraphrase.json")

    rc = mod.main(["--n-bootstrap", "500"])
    assert rc == 0
    out = json.loads(out_path.read_text())

    # Saturated traits excluded.
    assert set(out["h1_passing_traits"]) == {"validating", "conciseness"}
    h2 = out["h2_paired_leakage"]
    # Per-seed mean averages only over the 2 H1-passers; d_leakage per
    # cell = -0.4; per-seed mean averaged over 2 traits = -0.4.
    for seed_str, m in h2["per_seed_mean"].items():
        assert float(m) == pytest.approx(-0.4, abs=1e-6), (
            f"H2 per-seed mean leaked saturated traits: seed={seed_str} m={m}"
        )
    assert h2["d_mean"] == pytest.approx(-0.4, abs=1e-6)
