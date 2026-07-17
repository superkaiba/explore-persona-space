"""#1434 pins: D0 DVSpec amendment, bank wiring, round registration seams,
verdict/lattice gates, margin-pool floor branch, errorbar-offset clamp."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from explore_persona_space.artifacts.behavior import BEHAVIORS, DVSpec  # noqa: E402


def test_writing_style_dvspec_amendment():
    """Plan §4 D0: judged_rate primary + tf_margin companion (impolite shape)."""
    b = BEHAVIORS["writing_style"]
    assert b.dv == DVSpec("judged_rate", "tf_margin")
    b.validate()  # never raises on the amended registration


def test_writing_style_bank_wiring():
    """Neutral bank 20/20 split, train/eval disjoint, extraction slot empty
    (the datagen-only-adoption carve-out; the pv driver reads the train bank)."""
    b = BEHAVIORS["writing_style"]
    assert len(b.train_question_bank) == 20
    assert len(b.eval_question_bank) == 20
    assert not (set(b.train_question_bank) & set(b.eval_question_bank))
    assert b.extraction is not None and b.extraction.question_set == ()
    bank = json.loads(
        (
            REPO / "src/explore_persona_space/artifacts/query_banks/writing_style_neutral_v1.json"
        ).read_text()
    )
    assert tuple(bank[0:20]) == b.train_question_bank
    assert tuple(bank[20:40]) == b.eval_question_bank


def test_questiongen_description_matches_registry():
    import issue1434_questiongen as qg1434

    assert qg1434.WRITING_STYLE_SPEC.description == BEHAVIORS["writing_style"].description


def test_pv_rubric_verbatim_and_provenance():
    """The committed rubric derives byte-for-byte from the committed paper
    template (only the two designed trait slots filled; runtime slots kept)."""
    rubric_path = (
        REPO
        / "src/explore_persona_space/artifacts/judge_prompts/pv_writing_style_trait_score_v1.txt"
    )
    text = rubric_path.read_text()
    assert "{question}" in text and "{answer}" in text
    assert "REFUSAL if the model clearly refused to answer" in text
    prov = json.loads(
        (rubric_path.parent / "pv_writing_style_trait_score_v1.provenance.json").read_text()
    )
    tmpl = (REPO / "scripts/issue1090_assets/pv_generation_prompt_template.txt").read_text()
    assert prov["template_sha256"] == hashlib.sha256(tmpl.encode()).hexdigest()
    assert prov["rubric_sha256"] == hashlib.sha256(text.encode()).hexdigest()


def test_round_registration_and_worker_cmd_seam():
    """The i1434 round registers with its own issue id + worker entrypoint, and
    the fu4 dispatcher's worker cmd routes subprocesses through issue1434_worker
    with --round i1434 (the external-round seam — a worker spawned through the
    fu4 script itself would die on 'unknown round')."""
    import argparse

    import issue1090_fu4 as fu4
    import issue1434_cells as cells

    spec = cells.register_i1434_round()
    assert spec.issue == 1434
    assert spec.upload_all_rungs is True
    assert len(cells.I1434_RUNS) == 12
    assert {r.cell_key for r in cells.I1434_RUNS} == set(cells.CELL_KEYS)
    assert cells.RUN_BY_ID_1434["ws-icl-lr1e4"].run_name == "issue1434_ws-icl-lr1e4_seed42"
    prior = fu4.ROUND
    try:
        fu4.set_round("i1434")
        args = argparse.Namespace(
            smoke=True,
            round="i1434",
            out_root_resolved="/tmp/x",
            sentinel_dir_resolved="/tmp/y",
            seed=42,
            upload=False,
            manifest=None,
            eval_question_limit=None,
        )
        cmd = fu4._worker_cmd(args, cells.I1434_RUNS[0], slot=0)
        assert cmd[3].endswith("issue1434_worker.py"), cmd[3]
        assert "--round" in cmd and cmd[cmd.index("--round") + 1] == "i1434"
        # fu4/fu5 defaults stay byte-identical (regression on the new fields)
        assert fu4.ROUNDS["fu4"].issue == 1090
        assert fu4.ROUNDS["fu4"].worker_script == ""
        assert fu4.ROUNDS["fu4"].upload_all_rungs is False
        assert fu4.FU4_RUNS[0].run_name == "issue1090_fu4_fmt-pers-lr1e5_seed42"
    finally:
        fu4.ROUND = prior


def test_smoke_resolver_is_one_run_production_grid_is_twelve():
    import issue1090_fu4 as fu4
    import issue1434_cells as cells

    cells.register_i1434_round()
    prior = fu4.ROUND
    try:
        fu4.set_round("i1434")
        assert [r.run_id for r in fu4.resolve_fu4_runs(None, smoke=True)] == ["ws-pers-lr1e5"]
        assert len(fu4.resolve_fu4_runs(None, smoke=False)) == 12
    finally:
        fu4.ROUND = prior


def test_verdict_arm_rule_branches():
    import issue1434_cells as cells

    sels = {
        f"ws-bare-{t}": {"in_band": False, "rate": r, "step": 5}
        for t, r in [("lr1e5", 0.10), ("lr3e5", 0.62), ("lr1e4", 0.70)]
    }
    sels["ws-bare-lr3e5"]["in_band"] = True
    sels["ws-bare-lr1e4"]["in_band"] = True
    rid, rec = cells.verdict_arm_for_context("ws-bare", sels)
    assert rid == "ws-bare-lr3e5" and rec["rule"] == "lowest_lr_in_band"  # LOWEST lr wins
    for k in sels:
        sels[k]["in_band"] = False
    rid, rec = cells.verdict_arm_for_context("ws-bare", sels)
    assert rid == "ws-bare-lr3e5" and rec["rule"] == "closest_approach"
    with pytest.raises(ValueError, match="missing selections"):
        cells.verdict_arm_for_context("ws-bare", {"ws-bare-lr1e5": sels["ws-bare-lr1e5"]})


def test_lattice_verdict_disjoint_exhaustive():
    import issue1434_cells as cells

    assert cells.lattice_verdict(0.0, (-0.1, 0.2)) == "Installed"  # q_band >= 0
    assert cells.lattice_verdict(-0.01, (0.05, 0.2)) == "Dose-responsive-but-short"
    assert cells.lattice_verdict(-0.01, (-0.05, 0.2)) == "Not-installed"
    assert cells.lattice_verdict(-0.01, (0.0, 0.2)) == "Not-installed"  # CI touching 0


def test_margin_pool_floor_branch(tmp_path, monkeypatch):
    """The A13-parity ship-without-margin escape fires below 15/15 (degenerate
    probe through the REAL derive path), and a healthy pool sha-pins."""
    import issue1434_cells as cells

    d = tmp_path / "datagen_cells" / "ws-pers" / "datagen"
    d.mkdir(parents=True)

    def _write(n_pos: int, n_neg: int) -> None:
        rows_p, rows_n, judged = [], [], []
        for i in range(n_pos):
            rows_p.append(
                {
                    "request_id": f"p{i}",
                    "arm": "positive",
                    "question": f"q{i}",
                    "completion": f"hey {i}",
                    "question_id": i,
                    "variant_id": 0,
                }
            )
            judged.append({"request_id": f"p{i}", "kept": True})
        for i in range(n_neg):
            rows_n.append(
                {
                    "request_id": f"n{i}",
                    "arm": "negative",
                    "question": f"q{i}",
                    "completion": f"Indeed {i}.",
                    "question_id": i,
                    "variant_id": 0,
                }
            )
            judged.append({"request_id": f"n{i}", "kept": True})
        (d / "raw_pos.jsonl").write_text("\n".join(json.dumps(r) for r in rows_p) + "\n")
        (d / "raw_neg.jsonl").write_text("\n".join(json.dumps(r) for r in rows_n) + "\n")
        (d / "judge_rows.jsonl").write_text("\n".join(json.dumps(r) for r in judged) + "\n")

    class Cfg:
        out_root = tmp_path

    _write(5, 20)  # pos below the 15 floor
    pos, neg, meta = cells.i1434_margin_pools(Cfg())
    assert pos is None and meta["status"] == "skipped_pool_below_floor"
    _write(30, 18)  # healthy: equalize down to 18, cap 25
    pos, neg, meta = cells.i1434_margin_pools(Cfg())
    assert len(pos) == len(neg) == 18
    assert meta["pool_sha256"] and meta["equalized_to"] == 18


def test_figures_errorbar_offsets_clamp_inverted_ci(tmp_path):
    """#547/#1335: a tiny-n INVERTED CI routes through the REAL figure fn to
    savefig without a matplotlib negative-yerr ValueError."""
    import issue1434_figures as figs

    lo, hi = figs._err(0.5, [0.6, 0.4])  # inverted CI
    assert lo == 0.0 and hi == 0.0
    agg = {
        "band": [0.60, 0.85],
        "tier2": {
            "ws-pers": {
                "base": {"rate": 0.5, "wilson_95": [0.62, 0.41]},  # inverted on purpose
                "verdict_arm": {"run_id": "ws-pers-lr1e5"},
                "trained": {"rate": 0.7, "wilson_95": [0.75, 0.6]},
            }
        },
        "ladders": {},
    }
    out = figs.fig_install_grid(agg, tmp_path)
    assert out.exists() and out.stat().st_size > 0


def test_worker_delegation_routes_fu4_phases(monkeypatch):
    """--phase dispatch/run forwards VERBATIM to fu4.main with --round i1434
    injected (the round is registered before delegation)."""
    import issue1090_fu4 as fu4
    import issue1434_worker as w

    seen: dict = {}

    def fake_main(argv):
        seen["argv"] = list(argv)
        return 0

    monkeypatch.setattr(fu4, "main", fake_main)
    rc = w.main(["--smoke", "--phase", "dispatch", "--dry-run"])
    assert rc == 0
    assert seen["argv"][:2] == ["--round", "i1434"]
    assert "--phase" in seen["argv"] and "dispatch" in seen["argv"]


def test_judge_dispatch_bare_scalar_score_not_erased():
    """#1434 fix (the #778 scalar-passthrough, dispatch-path parity): a judge
    that answers the bare integer '95' (the persona-vectors rubric's own
    'just the number' instruction) must land as {'score': 95} — NOT be erased
    to a parse_error drop by the dict-shaped result plumbing. Fails pre-fix
    (every parse site returned parse_error for scalar responses; the #1434
    extract smoke dropped 4/4 exhibit-arm rollouts to it)."""
    from explore_persona_space.eval.judge_dispatch import (
        _normalize_scalar_score,
        _parsed_with_raw,
    )
    from explore_persona_space.eval.utils import parse_judge_json

    parsed = _normalize_scalar_score(parse_judge_json("95"))
    assert parsed == {"score": 95}
    from explore_persona_space.eval.graded_judge import _score_from_parsed

    assert _score_from_parsed(parsed) == 95.0
    # envelope + REFUSAL + out-of-range dispositions unchanged
    assert _normalize_scalar_score({"score": 40}) == {"score": 40}
    assert _normalize_scalar_score(parse_judge_json("250")) == 250  # out-of-range: caller drops
    assert _score_from_parsed(250) is None
    assert _normalize_scalar_score(True) is True  # bool never a score
    # non-dict parses must pass through raw-retention untouched (dict(95) crashed)
    assert _parsed_with_raw(95, "95") == 95


def test_install_grid_covers_all_trained_arms():
    """Round-2 Major-3 pin (plan D5 'second grid'): the own-context install grid
    carries EVERY trained arm in proj.states (12 in production), not only the
    verdict arms — verdict arm via Tier-2 delta, non-verdict arms via their
    selection-rung Tier-1 rate minus the per-context Tier-2 base rate."""
    import issue1434_cells as cells
    import issue1434_pv as pv

    cells.register_i1434_round()
    run_ids = [r.run_id for r in cells.I1434_RUNS if r.cell_key == "ws-pers"]
    assert len(run_ids) == 3
    verdict_run = run_ids[0]
    aggregate = {
        "ladders": {
            rid: {"status": "trained", "selection": {"rate": 0.60 + 0.05 * i}}
            for i, rid in enumerate(run_ids)
        },
        "tier2": {
            "ws-pers": {
                "verdict_arm": {"run_id": verdict_run},
                "base": {"rate": 0.10},
                "delta": 0.55,
            }
        },
        "panel": {},
    }
    proj = {"states": {rid: {} for rid in run_ids}}
    grids = pv._cell_grids(None, aggregate, proj)
    inst = grids["install"]
    assert sorted(inst["state_ids"]) == sorted(run_ids)  # n_states == n_trained_runs
    by_state = dict(
        zip(inst["state_ids"], zip(inst["y"], inst["y_basis"], strict=True), strict=True)
    )
    assert by_state[verdict_run] == (0.55, "tier2_delta")
    for i, rid in enumerate(run_ids):
        if rid == verdict_run:
            continue
        y, basis = by_state[rid]
        assert basis == "tier1_selection_rate_minus_tier2_base_rate"
        assert y == pytest.approx((0.60 + 0.05 * i) - 0.10)


def test_spearman_signed_twin_and_abs_consistency():
    """Round-2 Major-2 pin: the SIGNED Spearman helper preserves sign (the H3
    'CI excludes 0 on the positive side' verdict input); the |rho| twin is its
    elementwise absolute value (the selection/null-band statistic)."""
    import issue1434_pv as pv
    import numpy as np

    y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    P = np.stack([y, -y, np.array([2.0, 1.0, 4.0, 3.0, 5.0])])
    signed = pv._spearman_signed_per_layer(P, y)
    assert signed[0] == pytest.approx(1.0)
    assert signed[1] == pytest.approx(-1.0)  # a NEGATIVE rho survives (no abs)
    assert 0.0 < signed[2] < 1.0
    np.testing.assert_allclose(pv._spearman_obs_per_layer(P, y), np.abs(signed))


def test_tier2_lattice_none_propagation():
    """Round-2 Minor-4 pin (drop-never-coerce): an all-dropped arm (rate None)
    propagates None through q_band/delta/CI and reads
    'not_computable_all_dropped' — never a lattice verdict from a coerced 0.0;
    the normal path still computes the registered fields."""
    import issue1434_worker as w

    dropped = {"tag": "t2-trained-x", "rate": None, "k_positive": 0, "n_scored": 0}
    base = {"tag": "t2-base-x", "rate": 0.10, "k_positive": 20, "n_scored": 200}
    out = w._tier2_lattice_fields(dropped, base)
    assert out == {
        "q_band": None,
        "delta": None,
        "delta_newcombe_95": None,
        "lattice_verdict": "not_computable_all_dropped",
    }
    trained = {"tag": "t2-trained-y", "rate": 0.70, "k_positive": 140, "n_scored": 200}
    ok = w._tier2_lattice_fields(trained, base)
    assert ok["q_band"] == pytest.approx(0.70 - 0.60)
    assert ok["delta"] == pytest.approx(0.60)
    assert len(ok["delta_newcombe_95"]) == 2
    assert isinstance(ok["lattice_verdict"], str)
    assert ok["lattice_verdict"] != "not_computable_all_dropped"


def test_phase_validate_battery_signed_bootstrap(tmp_path):
    """Round-2 Major-2/Minor-7 end-to-end probe: phase_validate's battery on a
    REAL synthetic 3-state root (past the insufficient_cells gate the 1-run
    smoke takes) persists the SIGNED observed vector, a SIGNED frozen-layer
    cluster bootstrap ('CI excludes 0 on the positive side' readable), the
    selection-INHERITED signed bootstrap, and the per-draw signed matrix."""
    import argparse

    import issue1434_cells as cells
    import issue1434_pv as pv
    import numpy as np
    import torch

    from explore_persona_space.artifacts.directions import DirectionResult, save_direction

    rng = np.random.default_rng(7)
    layers, hidden = (0, 1), 6
    root = tmp_path / "pv"
    save_direction(
        DirectionResult(
            behavior_name="writing_style",
            regime="read_out",
            layers=layers,
            r_b=torch.randn(2, hidden, generator=torch.Generator().manual_seed(0)),
            counts={},
            provenance="on_policy",
        ),
        root / "rb_writing_style.pt",
    )
    torch.save(
        {"exhibit": torch.randn(10, 2, hidden), "not_exhibit": torch.randn(10, 2, hidden)},
        root / "extraction_pools.pt",
    )
    run_ids = [r.run_id for r in cells.I1434_RUNS if r.cell_key == "ws-pers"]
    proj_states = {}
    cap_root = root / "capture"
    base_means = {arm: torch.randn(2, hidden) for arm in pv.CAPTURE_ARMS}
    torch.save({"cell_key": "ws-pers", "means": base_means}, _mk(cap_root / "base-ws-pers"))
    for i, rid in enumerate(run_ids):
        means = {arm: base_means[arm] + (i + 1) * 0.5 for arm in pv.CAPTURE_ARMS}
        torch.save({"cell_key": "ws-pers", "means": means}, _mk(cap_root / rid))
        proj_states[rid] = {
            arm: {
                "projection": [float(i) + rng.normal(0, 0.05), float(i) + rng.normal(0, 0.05)],
                "cosine": [0.5, 0.5],
                "shift_norm": [1.0, 1.0],
            }
            for arm in pv.CAPTURE_ARMS
        }
    import issue1090_run as run1090

    run1090._atomic_write_json(
        root / "projections.json",
        {"layers": list(layers), "arms": list(pv.CAPTURE_ARMS), "states": proj_states},
    )
    deliver = tmp_path / "deliverables"
    deliver.mkdir()
    run1090._atomic_write_json(
        deliver / "i1434_ladders.json",
        {
            "ladders": {
                rid: {"status": "trained", "selection": {"rate": 0.60 + 0.1 * i}}
                for i, rid in enumerate(run_ids)
            },
            "tier2": {
                "ws-pers": {
                    "verdict_arm": {"run_id": run_ids[0]},
                    "base": {"rate": 0.10},
                    "delta": 0.50,
                }
            },
            "panel": {},
        },
    )
    cfg = argparse.Namespace(out_root=tmp_path, smoke=True, upload=False)
    assert pv.phase_validate(cfg, argparse.Namespace()) == 0
    out = json.loads((deliver / "pv_validation.json").read_text())
    grid = out["grids"]["install"]
    assert grid["n_states"] == 3 and grid["n_cells"] == 3  # Major-3 full-arm grid
    assert len(grid["y_basis"]) == 3
    arm = grid["response_shared"]
    assert len(arm["observed_signed_rho_per_layer"]) == 2
    boot = arm["cluster_bootstrap_headline_layer"]
    assert boot["statistic"] == "signed_rho_frozen_full_sample_headline_layer"
    assert boot["p2_5"] <= boot["p97_5"]
    assert isinstance(boot["ci_excludes_zero_positive"], bool)
    inh = arm["cluster_bootstrap_selection_inherited"]
    assert inh["statistic"] == "signed_rho_at_per_draw_max_abs_layer"
    matrix = json.loads(Path(boot["matrix"]).read_text())
    assert len(matrix["signed_rho_draws"][0]) == 2  # per-draw x per-layer SIGNED


def _mk(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d / "summary.pt"


# ── D1 top-up tranche (round 3: datagen-topup-tranche-not-wired concern) ─────
#
# Library seam (datagen.TopupSpec) + worker wiring (phase_datagen). Fakes ONLY
# at the external API boundary (the library's own GenerateFn/JudgeFn seams —
# real GenCandidate/JudgeResult types, signature-conformant), per the
# one-production-body-test rule; every other stage (schedule composition,
# judge-filter, floor check, emit, negative arm, mix assembly) runs its REAL
# body. Fails-pre-fix: generate_training_data had no ``topup`` kwarg and a
# near-miss cell took the G1 drop path.

from explore_persona_space.artifacts import datagen as _dg  # noqa: E402
from explore_persona_space.artifacts.context import context_for_persona  # noqa: E402
from explore_persona_space.artifacts.datagen import GenCandidate, TopupSpec  # noqa: E402
from explore_persona_space.eval.graded_judge import JudgeResult  # noqa: E402

_TOPUP_SRC = context_for_persona("villain")  # disjoint from the default panel
_TOPUP_BEH = BEHAVIORS["sycophancy"]  # no structural predicate; parent-fixture parity


def _gen_all_topup():
    def gen(requests):
        return [GenCandidate(r, f"resp::{r.request_id}") for r in requests]

    return gen


def _judge_first_n(keep_first_n: int, *, keep_topup: bool = True):
    """Boundary judge stub forcing a first-sample near miss: keeps exactly
    ``keep_first_n`` first-sample positives (``pos-*``, first-seen order),
    every tranche positive (``tpos-*``) iff ``keep_topup``, and every
    negative (the negative keep rule is score < threshold)."""
    kept_first: set[str] = set()

    def judge(
        items,
        eval_prompt,
        *,
        n_draws,
        cache_dir,
        save_raw,
        judge_model,
        dry_run=False,
        max_tokens=64,
    ):
        scores = {}
        for rid, _q, _c in items:
            if rid.startswith("tpos-"):
                scores[rid] = 80.0 if keep_topup else 20.0
            elif rid.startswith("pos-"):
                if rid in kept_first or len(kept_first) < keep_first_n:
                    kept_first.add(rid)
                    scores[rid] = 80.0
                else:
                    scores[rid] = 20.0
            else:
                scores[rid] = 20.0
        return JudgeResult(
            scores=scores,
            n_total_draws=len(items) * n_draws,
            n_dropped_draws=0,
            per_item_draw_counts={rid: n_draws for rid, _, _ in items},
            per_item_scores={rid: [scores[rid]] * n_draws for rid, _, _ in items},
        )

    return judge


def test_datagen_topup_default_off_unchanged_and_no_fire_on_healthy(tmp_path):
    """Parent-caller parity: no ``topup`` -> no tranche artifacts; an ARMED
    topup on a healthy cell (kept >= target) never fires; the manifest resume
    key is IDENTICAL with and without topup (a near-miss re-enters the same
    regime and replays the first sample from cache)."""
    common = dict(
        target_n=6,
        n_judge_draws=2,
        generate_fn=_gen_all_topup(),
        judge_fn=_judge_first_n(10_000),  # healthy: keep every positive
    )
    _dg.generate_training_data(
        _TOPUP_BEH, _TOPUP_SRC, "default_v1", out_dir=tmp_path / "off", **common
    )
    _dg.generate_training_data(
        _TOPUP_BEH,
        _TOPUP_SRC,
        "default_v1",
        out_dir=tmp_path / "on",
        topup=TopupSpec(),
        **common,
    )
    for d in (tmp_path / "off", tmp_path / "on"):
        assert not (d / "raw_pos_topup.jsonl").exists()
        assert not (d / "topup_record.json").exists()
        meta = json.loads((d / "pool_meta.json").read_text())
        assert "topup" not in meta
    m_off = json.loads((tmp_path / "off" / "gen_manifest.json").read_text())
    m_on = json.loads((tmp_path / "on" / "gen_manifest.json").read_text())
    assert m_off == m_on  # topup never enters the resume key


def test_datagen_topup_tranche_fires_and_yield_dv_frozen(tmp_path):
    """FAILS PRE-FIX (no ``topup`` kwarg). Near miss (kept 3 < floor 5 at
    target 6) -> the ONE tranche fires, the union clears the floor, emit +
    negative + mix inputs proceed; the yield DV (pool_meta positive arm +
    judge_rows.jsonl) stays FROZEN at the first sample; the tranche is
    recorded separately with question_id-dedupe accounting."""
    out = tmp_path / "cell"
    _dg.generate_training_data(
        _TOPUP_BEH,
        _TOPUP_SRC,
        "default_v1",
        out_dir=out,
        target_n=6,  # floor_n = 5
        n_judge_draws=2,
        generate_fn=_gen_all_topup(),
        judge_fn=_judge_first_n(3, keep_topup=True),
        topup=TopupSpec(),
    )
    meta = json.loads((out / "pool_meta.json").read_text())
    assert meta["positive"]["kept"] == 3  # FROZEN first-sample yield DV
    top = meta["topup"]
    assert top["fired"] is True and top["union_floor_missed"] is False
    assert top["kept_pos_first_sample"] == 3
    assert top["tranche_requested"] == 9  # ceil(6 / EXPECTED_YIELD)
    assert top["tranche_kept"] == top["tranche_merged"] + top["tranche_dedup_dropped_qid"]
    assert top["kept_pos_union"] == 3 + top["tranche_merged"] >= 5
    # Merged tranche rows never duplicate a first-sample question id.
    rec = json.loads((out / "topup_record.json").read_text())
    assert rec["union_floor_missed"] is False
    assert (out / "raw_pos_topup.jsonl").exists()
    assert (out / "judge_rows_topup.jsonl").exists()
    # judge_rows.jsonl (the yield-DV sidecar) carries NO tranche rows.
    rows = [
        json.loads(line)
        for line in (out / "judge_rows.jsonl").read_text().split("\n")
        if line.strip()
    ]
    assert rows and not any(r["request_id"].startswith("tpos-") for r in rows)
    # Emit contract unchanged: exactly floor_n positives in pos.jsonl.
    pos_rows = [line for line in (out / "pos.jsonl").read_text().split("\n") if line.strip()]
    assert len(pos_rows) == 5


def test_datagen_topup_g1_after_tranche_and_second_attempt_refuses(tmp_path):
    """G1 runs AFTER the tranche: a union still below floor raises
    DatagenYieldError naming the tranche; the miss is durably recorded; a
    SECOND attempt on the same dir refuses loudly (EXACTLY ONE tranche)."""
    out = tmp_path / "cell"
    kwargs = dict(
        target_n=6,
        n_judge_draws=2,
        generate_fn=_gen_all_topup(),
        topup=TopupSpec(),
    )
    with pytest.raises(_dg.DatagenYieldError, match="AFTER the single allowed tranche"):
        _dg.generate_training_data(
            _TOPUP_BEH,
            _TOPUP_SRC,
            "default_v1",
            out_dir=out,
            judge_fn=_judge_first_n(0, keep_topup=False),
            **kwargs,
        )
    rec = json.loads((out / "topup_record.json").read_text())
    assert rec["union_floor_missed"] is True and rec["fired"] is True
    with pytest.raises(RuntimeError, match="EXACTLY ONE"):
        _dg.generate_training_data(
            _TOPUP_BEH,
            _TOPUP_SRC,
            "default_v1",
            out_dir=out,
            judge_fn=_judge_first_n(0, keep_topup=False),
            **kwargs,
        )


def test_datagen_topup_reuse_pos_refuses(tmp_path):
    """topup x reuse_pos is an untested combination -> loud ValueError (checked
    before any staged-file access, so dummy paths suffice)."""
    spec = _dg.PosReuseSpec(
        raw_pos_path=tmp_path / "nope.jsonl",
        judge_rows_path=tmp_path / "nope2.jsonl",
        expected_kept_count=1,
        provenance={},
    )
    with pytest.raises(ValueError, match="not supported together"):
        _dg.generate_training_data(
            _TOPUP_BEH,
            _TOPUP_SRC,
            "default_v1",
            out_dir=tmp_path / "x",
            target_n=6,
            generate_fn=_gen_all_topup(),
            judge_fn=_judge_first_n(10_000),
            reuse_pos=spec,
            topup=TopupSpec(),
        )


@pytest.mark.parametrize("cell_key", ["ws-pers", "ws-bare", "ws-conv", "ws-icl"])
def test_phase_datagen_topup_wiring_near_miss(tmp_path, monkeypatch, cell_key):
    """FAILS PRE-FIX (near miss ended ``yield_floor_missed``). The REAL
    ``phase_datagen`` body per ARM CLASS (persona / bare+filtered-panel /
    wildchat prefix / ICL prefix): a forced first-sample near miss (kept 3 <
    floor 5) is rescued by the ONE tranche -> status ``success`` with the
    tranche recorded separately, the yield DV frozen at the first sample, and
    the mix built by the UNCHANGED assembler. Fakes only at the external
    boundaries (Claude gen / Sonnet judge / HF corpus staging + provenance /
    tokenizer download)."""
    import argparse

    import issue1434_cells as c1434
    import issue1434_worker as w1434
    import transformers

    c1434.register_i1434_round()
    monkeypatch.setattr(_dg, "_default_generate_fn", lambda **kw: _gen_all_topup())
    monkeypatch.setattr(_dg, "judge_graded", _judge_first_n(3, keep_topup=True))

    def _stub_corpus(dest, **kw):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "prompt": [{"role": "user", "content": f"generic q {i}"}],
                "completion": [{"role": "assistant", "content": f"generic a {i}"}],
            }
            for i in range(200)
        ]
        dest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return str(dest)

    monkeypatch.setattr(w1434.i1074, "_stage_generic_corpus", _stub_corpus)
    monkeypatch.setattr(
        w1434, "_generic_corpus_provenance", lambda p: {"stub": True, "path": str(p)}
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: None),  # tokenizer=None: budget gate off (smoke seam)
    )
    args = argparse.Namespace(
        smoke=True,
        full=False,
        cells=cell_key,
        out_root=str(tmp_path / "root"),
        sentinel_dir=str(tmp_path / "logs"),
        seed=42,
        eval_question_limit=None,
        upload=False,
        oversample_mult=None,
    )
    cfg = w1434.worker_config(args)
    out = w1434.phase_datagen(cfg, args)
    rec = out[cell_key]
    assert rec["status"] == "success", rec
    assert rec["topup_considered"] is True
    top = rec["topup_record"]
    assert top["fired"] is True and top["kept_pos_first_sample"] == 3
    assert top["kept_pos_union"] >= 5  # floor cleared by the tranche
    meta = json.loads(Path(rec["pool_meta_path"]).read_text())
    assert meta["positive"]["kept"] == 3  # frozen yield DV
    assert meta["topup"]["tranche_requested"] == 9  # ceil(smoke target 6 / 0.7)
    assert rec["train_mix_sha256"]  # mix built (unchanged assembler)
    # Resume: the recorded cell (topup_considered) SKIPS — one tranche per cell.
    out2 = w1434.phase_datagen(cfg, args)
    assert out2[cell_key]["ts"] == rec["ts"]


def test_phase_datagen_pre_topup_miss_reenters_once(tmp_path, monkeypatch):
    """A legacy same-mult ``yield_floor_missed`` record WITHOUT the top-up
    lever (pre-wiring shape) re-enters ONCE and gets rescued; the new record
    carries ``topup_considered`` so the next invocation skips."""
    import argparse

    import issue1434_cells as c1434
    import issue1434_worker as w1434
    import transformers

    c1434.register_i1434_round()
    monkeypatch.setattr(_dg, "_default_generate_fn", lambda **kw: _gen_all_topup())
    monkeypatch.setattr(_dg, "judge_graded", _judge_first_n(3, keep_topup=True))

    def _stub_corpus(dest, **kw):
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "prompt": [{"role": "user", "content": f"generic q {i}"}],
                "completion": [{"role": "assistant", "content": f"generic a {i}"}],
            }
            for i in range(200)
        ]
        dest.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        return str(dest)

    monkeypatch.setattr(w1434.i1074, "_stage_generic_corpus", _stub_corpus)
    monkeypatch.setattr(
        w1434, "_generic_corpus_provenance", lambda p: {"stub": True, "path": str(p)}
    )
    monkeypatch.setattr(
        transformers.AutoTokenizer,
        "from_pretrained",
        classmethod(lambda cls, *a, **kw: None),
    )
    args = argparse.Namespace(
        smoke=True,
        full=False,
        cells="ws-pers",
        out_root=str(tmp_path / "root"),
        sentinel_dir=str(tmp_path / "logs"),
        seed=42,
        eval_question_limit=None,
        upload=False,
        oversample_mult=None,
    )
    cfg = w1434.worker_config(args)
    cell_root = cfg.out_root / "datagen_cells" / "ws-pers"
    cell_root.mkdir(parents=True, exist_ok=True)
    legacy = {
        "cell_key": "ws-pers",
        "status": "yield_floor_missed",
        "reason": "legacy pre-top-up miss",
        "oversample_mult": 2.5,  # fu3w.DEFAULT_OVERSAMPLE_MULT — same budget
    }
    (cell_root / "datagen_summary_1434.json").write_text(json.dumps(legacy))
    out = w1434.phase_datagen(cfg, args)
    rec = out["ws-pers"]
    assert rec["status"] == "success" and rec["topup_record"]["fired"] is True
