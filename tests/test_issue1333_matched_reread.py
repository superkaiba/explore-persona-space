"""#1333 matched-install-breadth-reread round pins (plan §4b/§4c/§4e/§4f).

1. The FOUR new dispatcher flags default to the parent constants — parse-args
   with no new flags reproduces the module constants and the EXACT pre-diff
   ``regime_key()`` dict (byte-identical behavior when absent).
2. ``regime_key()`` gains the TWO selection-affecting overrides when (and only
   when) they differ from the defaults — a stale parent-regime artifact can
   never vouch for this round's phases (plan §4b resume-provenance).
3. ``--save-steps-override`` rides the regime via ``Cfg.save_steps_for`` (the
   plan's "save_steps already rides regime_key via save_steps_for" claim was
   true only for the CONSTANT — the Cfg-level accessor closes the gap).
4. ``--ladder-read-steps`` restricts/reorders p3 reads and fails loud on a
   step with no persisted rung.
5. The analysis script's §3 paired statistic + question-cluster bootstrap +
   DISJOINT verdict lattice, on synthetic per_probe fixtures with known ground
   truth (all three lattice branches).
6. The §4f calibration-offset caveat flag fires at |offset| > 0.5 nat.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

from explore_persona_space.experiments import issue_1333 as C

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _dispatch():
    import issue1333_dispatch as d

    return d


def _analysis():
    import issue1333_matched_reread_analysis as a

    return a


def _legacy_regime(cells: tuple[str, ...], *, smoke: bool = False) -> dict:
    """The EXACT pre-diff regime_key dict, rebuilt from module constants."""
    return {
        "issue": C.ISSUE,
        "smoke": smoke,
        "cells": list(cells),
        "seed": C.SEED,
        "eval_question_limit": None if not smoke else 2,
        "window": list(C.ACCEPT_WINDOW),
        "lora_max_steps": C.LORA_MAX_STEPS,
        "save_steps": {c: C.save_steps_for(c) for c in C.NEW_LORA_CELLS},
        "warmup_steps": C.LORA_WARMUP_STEPS,
        "ft_grid": list(C.FT_GRID),
    }


# ── 1/2/3: flag defaults + regime keying ─────────────────────────────────────


def test_flag_defaults_byte_identical(tmp_path):
    d = _dispatch()
    args = d._parse_args(["--full", "--cells", C.CELL_EXT_BARE, "--out-root", str(tmp_path)])
    cfg = d.build_cfg(args)
    assert cfg.install_target == C.TARGET_DELTA_G
    assert cfg.accept_window_nats == C.MATCH_TOL_NATS
    assert cfg.accept_window == C.ACCEPT_WINDOW
    assert cfg.save_steps_override is None
    assert cfg.ladder_read_steps is None
    assert cfg.mixes_from_hub is False
    assert cfg.calibration_checkpoint is None
    assert cfg.selection_overridden is False
    assert cfg.hub_variant == ""
    # regime dict EXACTLY the pre-diff shape — no new keys at defaults
    assert cfg.regime_key() == _legacy_regime((C.CELL_EXT_BARE,))
    # per-cell cadence unchanged at defaults
    for cell in C.NEW_LORA_CELLS:
        assert cfg.save_steps_for(cell) == C.save_steps_for(cell)
    # default unit command lines carry NO new flags
    ua = d._unit_args(cfg, "ladder", C.CELL_EXT_BARE)
    for flag in (
        "--install-target",
        "--accept-window-nats",
        "--save-steps-override",
        "--ladder-read-steps",
        "--mixes-from-hub",
        "--calibration-checkpoint",
    ):
        assert flag not in ua, flag


def test_regime_key_changes_on_selection_overrides(tmp_path):
    d = _dispatch()
    base = d.Cfg(smoke=False, cells=(C.CELL_EXT_BARE,), out_root=tmp_path)
    over = d.Cfg(
        smoke=False,
        cells=(C.CELL_EXT_BARE,),
        out_root=tmp_path,
        install_target=5.5141,
        accept_window_nats=1.0,
    )
    assert base.regime_key() == _legacy_regime((C.CELL_EXT_BARE,))
    rk = over.regime_key()
    assert rk != base.regime_key()
    assert rk["install_target"] == 5.5141
    assert rk["accept_window_nats"] == 1.0
    assert rk["window"] == [4.5141, 6.5141]
    assert over.selection_overridden is True
    assert over.hub_variant == "matched55"
    # forwarded to fanout units (repr round-trips floats exactly)
    ua = d._unit_args(over, "ladder", C.CELL_EXT_BARE)
    assert ua[ua.index("--install-target") + 1] == repr(5.5141)
    assert float(ua[ua.index("--install-target") + 1]) == 5.5141


def test_save_steps_override_rides_regime(tmp_path):
    d = _dispatch()
    cfg = d.Cfg(smoke=False, cells=(C.CELL_EXT_BARE,), out_root=tmp_path, save_steps_override=5)
    assert cfg.save_steps_for(C.CELL_EXT_BARE) == 5
    rk = cfg.regime_key()
    assert rk["save_steps"][C.CELL_EXT_BARE] == 5
    assert rk != _legacy_regime((C.CELL_EXT_BARE,))
    # NO selection-override keys ride a cadence-only override
    assert "install_target" not in rk


# ── 4: --ladder-read-steps ───────────────────────────────────────────────────


def _fake_rungs(steps):
    return {s: Path(f"/tmp/rung-{s}") for s in steps}


def test_ladder_read_steps_restricts_and_reorders(tmp_path):
    d = _dispatch()
    rungs = _fake_rungs(range(5, 405, 5))
    cfg = d.Cfg(
        smoke=False,
        cells=(C.CELL_EXT_BARE,),
        out_root=tmp_path,
        ladder_read_steps=(95, 75, 80, 85, 90, 100),
    )
    pending = d._ladder_read_steps(cfg, C.CELL_EXT_BARE, rungs, {})
    assert pending == [95, 75, 80, 85, 90, 100]  # CSV order preserved (reorder)
    ladder = {95: {}, 75: {}}
    assert d._ladder_read_steps(cfg, C.CELL_EXT_BARE, rungs, ladder) == [80, 85, 90, 100]
    done = {s: {} for s in (75, 80, 85, 90, 95, 100)}
    assert d._ladder_read_steps(cfg, C.CELL_EXT_BARE, rungs, done) == []


def test_ladder_read_steps_missing_rung_fails_loud(tmp_path):
    d = _dispatch()
    rungs = _fake_rungs(range(10, 410, 10))  # cadence 10: step 85 has no rung
    cfg = d.Cfg(
        smoke=False,
        cells=(C.CELL_EXT_BARE,),
        out_root=tmp_path,
        ladder_read_steps=(80, 85, 90),
    )
    with pytest.raises(RuntimeError, match=r"\[85\] have no persisted rung"):
        d._ladder_read_steps(cfg, C.CELL_EXT_BARE, rungs, {})


def test_ladder_read_steps_default_schedule_unchanged(tmp_path):
    d = _dispatch()
    rungs = _fake_rungs(range(10, 410, 10))
    cfg = d.Cfg(smoke=False, cells=(C.CELL_EXT_BARE,), out_root=tmp_path)
    assert d._ladder_read_steps(cfg, C.CELL_EXT_BARE, rungs, {}) == C.coarse_read_steps(
        C.CELL_EXT_BARE, sorted(rungs)
    )


# ── selection threading values (the plan-§4a bracket) ───────────────────────


def test_select_rung_under_round_target_and_window():
    ladder = {
        80: {"delta_logp_mean": 4.407, "source_emission_rate": 0.0},
        85: {"delta_logp_mean": 5.1, "source_emission_rate": 0.0},
        90: {"delta_logp_mean": 5.8, "source_emission_rate": 0.0},
        95: {"delta_logp_mean": 6.4, "source_emission_rate": 0.0},
        100: {"delta_logp_mean": 7.116, "source_emission_rate": 0.0},
    }
    sel = C.select_rung(ladder, target=5.5141, window=(4.5141, 6.5141))
    # 90 (|5.8-5.5141|=0.286) beats 85 (0.414) and 95 (0.886); 80/100 out-of-window
    assert sel["step"] == 90 and sel["in_window"] is True


# ── 5/6: analysis fixtures — paired statistic, lattice, calibration ──────────

TRIO = ("chef", "hero", "philosopher")
N_Q = 20


def _slot_rec(
    deltas_by_label: dict[str, np.ndarray],
    *,
    source_delta: float,
    source_context_id: str,
    n_q: int = N_Q,
    extra_context: dict | None = None,
) -> dict:
    """Synthetic breadth slot_reads record mirroring run_breadth_unit's schema
    (per_probe rows with row-meta + trained/base four-float; per_context)."""
    base_logp, base_zm, base_ze, base_lz = -20.0, -8.0, 6.0, 1.5
    per_probe, per_context = [], {}
    all_labels = {"__source__": np.full(n_q, source_delta), **deltas_by_label}
    for label, deltas in all_labels.items():
        deltas = np.asarray(deltas, dtype=float)
        assert deltas.shape == (n_q,)
        cid = source_context_id if label == "__source__" else f"persona_{label}"
        for q in range(n_q):
            d = float(deltas[q])
            per_probe.append(
                {
                    "row": {"label": label, "context_id": cid, "q": q, "gen_emitted": False},
                    "trained": {
                        "logp": base_logp + d,
                        "z_marker": base_zm + d,  # margin delta == logp delta here
                        "z_eos": base_ze,
                        "logZ": base_lz,
                        "argmax_id": 0,
                    },
                    "base": {
                        "logp": base_logp,
                        "z_marker": base_zm,
                        "z_eos": base_ze,
                        "logZ": base_lz,
                        "argmax_id": 0,
                    },
                }
            )
        per_context[label] = {
            "context_id": cid,
            "delta_logp_mean": float(deltas.mean()),
            "emission_rate": 0.0,
            "base_prior_mean": base_logp,
        }
    if extra_context:
        per_context.update(extra_context)
    return {
        "n_probes": len(per_probe),
        "delta_logp_mean": float(
            np.mean([r["trained"]["logp"] - r["base"]["logp"] for r in per_probe])
        ),
        "source_emission_rate": 0.0,
        "per_probe": per_probe,
        "per_context": per_context,
        "cell": "synthetic",
        "panel_labels": list(all_labels),
    }


def _fixture_pair(gap: float, noise_sd: float = 0.05, seed: int = 7):
    """Matched + comparator records whose per-(context,question) paired
    difference has KNOWN mean ``gap`` (+ small noise)."""
    rng = np.random.default_rng(seed)
    villain = {t: 2.0 + rng.normal(0, noise_sd, N_Q) for t in TRIO}
    matched = {t: villain[t] + gap + rng.normal(0, noise_sd, N_Q) for t in TRIO}
    matched_rec = _slot_rec(matched, source_delta=5.5, source_context_id="bare_default")
    comparator_rec = _slot_rec(
        villain,
        source_delta=5.418,
        source_context_id="persona_villain",
        extra_context={
            "qwen_default": {
                "context_id": "persona_qwen_default",
                "delta_logp_mean": 0.5,
                "emission_rate": 0.0,
                "base_prior_mean": -20.0,
            }
        },
    )
    return matched_rec, comparator_rec


def _cal_rec(delta: float) -> dict:
    return {"delta_logp_mean": delta}


@pytest.mark.parametrize(
    ("gap", "expected"),
    [(2.0, "Survives"), (-2.0, "Dose-artifact"), (0.0, "Inconclusive")],
)
def test_paired_statistic_and_verdict_lattice(gap, expected):
    a = _analysis()
    matched, comparator = _fixture_pair(gap)
    out = a.build_comparison(matched, comparator, _cal_rec(5.514121723175049))
    assert out["primary"]["n_pairs"] == len(TRIO) * N_Q == 60
    assert out["primary"]["delta_mean"] == pytest.approx(gap, abs=0.05)
    lo, hi = out["primary"]["ci95"]
    assert lo < out["primary"]["delta_mean"] < hi
    assert out["primary"]["verdict"] == expected
    if expected == "Survives":
        assert lo > 0
    if expected == "Dose-artifact":
        assert hi < 0
    # secondaries present + descriptive
    tf = out["secondary_descriptive"]["margin_transfer_fractions"]
    assert set(tf["matched"]) == {*TRIO, "source_margin"}
    da = out["secondary_descriptive"]["default_rendered_asymmetry"]
    assert da["matched_label"] == "__source__"  # bare arm: default IS the source
    assert da["villain_label"] == "qwen_default"


def test_bootstrap_deterministic_and_vectorized_shape():
    a = _analysis()
    paired = np.random.default_rng(3).normal(1.0, 0.5, size=(3, N_Q))
    lo1, hi1, boot1 = a.question_cluster_bootstrap(paired)
    lo2, hi2, boot2 = a.question_cluster_bootstrap(paired)
    assert (lo1, hi1) == (lo2, hi2)  # seeded → deterministic
    assert boot1.shape == (a.BOOT_DRAWS,)
    assert np.array_equal(boot1, boot2)


def test_verdict_lattice_is_disjoint_and_exhaustive():
    a = _analysis()
    assert a.verdict(1.0, 0.2, 1.8) == "Survives"
    assert a.verdict(-1.0, -1.8, -0.2) == "Dose-artifact"
    assert a.verdict(0.1, -0.2, 0.4) == "Inconclusive"
    assert a.verdict(-0.1, -0.4, 0.2) == "Inconclusive"
    # boundary: CI touching 0 is NOT an exclusion
    assert a.verdict(0.5, 0.0, 1.0) == "Inconclusive"


def test_comparator_loader_assert_fails_loud():
    a = _analysis()
    matched, comparator = _fixture_pair(1.0)
    # (i) missing four-float key
    bad = {**comparator, "per_probe": [dict(r) for r in comparator["per_probe"]]}
    bad["per_probe"][0] = {
        **bad["per_probe"][0],
        "trained": {k: v for k, v in bad["per_probe"][0]["trained"].items() if k != "z_eos"},
    }
    with pytest.raises(ValueError, match="z_eos"):
        a.build_comparison(matched, bad, _cal_rec(5.5))
    # (ii) missing held-out label coverage
    thin = {
        **comparator,
        "per_probe": [r for r in comparator["per_probe"] if r["row"]["label"] != "chef"],
    }
    with pytest.raises(ValueError, match="chef"):
        a.build_comparison(matched, thin, _cal_rec(5.5))


def test_calibration_offset_caveat_fires_over_half_nat():
    a = _analysis()
    # 6.1 vs selection 5.5141 → offset +0.586 > 0.5 → caveat
    offsets, caveat = a.calibration_offsets(_cal_rec(6.1))
    assert caveat is True
    assert offsets["vs_parent_selection"] == pytest.approx(6.1 - 5.514121723175049)
    # 5.6 vs both refs (5.514, 5.418) → max |offset| 0.182 < 0.5 → no caveat
    offsets, caveat = a.calibration_offsets(_cal_rec(5.6))
    assert caveat is False
    # exact comparator-derived ref participates in the caveat
    _, caveat = a.calibration_offsets(_cal_rec(5.6), comparator_source_delta=5.0)
    assert caveat is True
