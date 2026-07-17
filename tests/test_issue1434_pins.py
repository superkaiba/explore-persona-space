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
