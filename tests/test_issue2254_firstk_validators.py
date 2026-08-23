"""Round-2/3 regression pins for the #2254 first-k driver (plan v10, r2+r3).

Covers the permanent invariants added after the r1 FAIL+FAIL verdict (task
#2254, `epm:code-review` v8) and the r3 reconciler round (`epm:code-review`
v9 + codex v2):

- producer-schema validators (`_validate_gen_record` / `_validate_judged_record`)
  reject truncated / mixed-grain / trace-less records BEFORE judge spend;
- the §3 denominator guard nulls the REGISTERED ratio points (`R`/`R1`/
  `R_span15` -> point None; raw ratio only under the diagnostic key);
- the per-cell validity gate (`_cell_validity`) enforces the rule-29
  completeness floor + coherence;
- the figures module skips validity-gated rows (`_row_valid`) and excludes
  "not-computable pending remediation" lattice blocks (`_lattice_blocks`);
- r3 BLOCKER firstk-pilot-pack-reachability: `phase_judge` uploads the pilot
  evidence pack on EVERY pilot-section exit (gate FAIL / §7 kill / --pilot
  return), mask-safe on the unwind path;
- r3 hardening: manifest-driven pack rehydration (stale-tail ignore,
  duplicate refusal, un-manifested refusal), local-first regime_fp
  cross-check, reduce-side judged-vintage gate, pilot-PASS input
  fingerprint, and the empty-draw regen-once validator-wedge escape.

tmp_path-only fixtures; no HF/network reads; no other issue's committed
eval_results (sparse-worktree safe).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import scripts.issue2254_first_k_steering as fk
import scripts.issue2254_firstk_figures as figs

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _cell() -> dict:
    return {
        "behavior": "evil",
        "direction": "rb",
        "breadth": "single",
        "position": "allans",
        "layer_config": "mid",
        "c": 2.0,
    }


def _gen_rec(cell: dict, cid: str, *, n_q: int = 3, n_draws: int = 2) -> dict:
    return {
        "cell_id": cid,
        "cell": cell,
        "q_of_context": [f"q{i}" for i in range(n_q)],
        "seeds": {
            "42": {
                "completions": [["text"] * n_draws for _ in range(n_q)],
                "edit_traces": [],
            }
        },
        "cap_hit_fraction": 0.0,
        "max_new_tokens": 2048,
    }


def _judged_rec(cell: dict, cid: str, *, n_q: int = 3) -> dict:
    return {
        "cell_id": cid,
        "cell": cell,
        "n_questions": n_q,
        "per_question_mean_score": [50.0] * n_q,
        "per_question_rate": [0.5] * n_q,
        "accounting": {"frac_items_complete": 1.0},
        "coherence_pass": True,
        "coherence_rate": 1.0,
    }


# ---------------------------------------------------------------------------
# producer-schema validators (judge/reduce inputs BEFORE spend)
# ---------------------------------------------------------------------------


def test_validate_gen_record_passes_on_valid() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    fk._validate_gen_record(_gen_rec(cell, cid), Path(f"{cid}.json"), n_q=3, n_draws=2)


def test_validate_gen_record_rejects_wrong_question_grain() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    rec = _gen_rec(cell, cid, n_q=2)  # truncated grain vs the invocation's 3
    with pytest.raises(AssertionError, match="q_of_context grain"):
        fk._validate_gen_record(rec, Path(f"{cid}.json"), n_q=3, n_draws=2)


def test_validate_gen_record_rejects_wrong_draw_count() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    rec = _gen_rec(cell, cid, n_draws=2)  # mixed-vintage 2-draw record vs 6
    with pytest.raises(AssertionError, match="draws != 6"):
        fk._validate_gen_record(rec, Path(f"{cid}.json"), n_q=3, n_draws=6)


def test_validate_gen_record_rejects_missing_edit_traces() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    rec = _gen_rec(cell, cid)
    del rec["seeds"]["42"]["edit_traces"]
    with pytest.raises(AssertionError):
        fk._validate_gen_record(rec, Path(f"{cid}.json"), n_q=3, n_draws=2)


def test_validate_gen_record_rejects_empty_completion() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    rec = _gen_rec(cell, cid)
    rec["seeds"]["42"]["completions"][1][0] = ""  # empty string draw
    with pytest.raises(AssertionError):
        fk._validate_gen_record(rec, Path(f"{cid}.json"), n_q=3, n_draws=2)


def test_validate_gen_record_rejects_identity_mismatch() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    rec = _gen_rec(cell, cid)
    rec["cell_id"] = "some__other__cell"
    with pytest.raises(AssertionError):
        fk._validate_gen_record(rec, Path(f"{cid}.json"), n_q=3, n_draws=2)


def test_validate_judged_record_passes_and_rejects_nq_mismatch() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    fk._validate_judged_record(_judged_rec(cell, cid), Path(f"{cid}.json"), n_q=3)
    bad = _judged_rec(cell, cid, n_q=2)  # judged at a different grain
    with pytest.raises(AssertionError):
        fk._validate_judged_record(bad, Path(f"{cid}.json"), n_q=3)


# ---------------------------------------------------------------------------
# §3 denominator guard: registered ratio points are None under the guard
# ---------------------------------------------------------------------------


def _lattice_inputs(a_off: float) -> tuple[dict, np.ndarray, dict]:
    nq = 20
    a0_q = np.linspace(40.0, 60.0, nq)
    arm_q = {
        "lastctx": a0_q + 0.1,
        "tok1": a0_q + 0.5,
        "span13": a0_q + 1.0,
        "span15": a0_q + 1.2,
        "allans": a0_q + a_off,
    }
    deg_q = {"allans": np.zeros(nq), "span13": np.zeros(nq)}
    return arm_q, a0_q, deg_q


def test_lattice_block_guard_nulls_registered_ratio_points() -> None:
    # Constant +2 all-answer delta: every resample |A_b| = 2 < the 5-point
    # floor -> unstable_frac = 1.0 -> ratio_unstable. Registered points None;
    # raw ratio only under the diagnostic key; verdict routes to Ambiguous.
    arm_q, a0_q, deg_q = _lattice_inputs(a_off=2.0)
    blk = fk._lattice_block("evil", "rb", "single", arm_q, a0_q, deg_q, "t-guard")
    assert blk["ratio_guard"]["ratio_unstable"] is True
    assert blk["R"]["point"] is None and blk["R"]["lo"] is None and blk["R"]["hi"] is None
    assert blk["R1"]["point"] is None
    assert blk["R_span15"]["point"] is None
    assert blk["R"]["raw_ratio_diagnostic_not_registered"] == pytest.approx(0.5)
    assert blk["verdict"] == "Ambiguous"
    # Descriptive fallback S - (2/3) A stays load-bearing under the guard.
    assert blk["fallback_S_minus_two_thirds_A"]["point"] == pytest.approx(1.0 - 2.0 * 2.0 / 3.0)


def test_lattice_block_unguarded_keeps_numeric_ratio_points() -> None:
    arm_q, a0_q, deg_q = _lattice_inputs(a_off=30.0)  # far above the 5-point floor
    blk = fk._lattice_block("evil", "rb", "single", arm_q, a0_q, deg_q, "t-clear")
    assert blk["ratio_guard"]["ratio_unstable"] is False
    assert blk["R"]["point"] == pytest.approx(1.0 / 30.0)
    assert blk["R"]["lo"] is not None and blk["R"]["hi"] is not None
    assert blk["R_span15"]["point"] == pytest.approx(1.2 / 30.0)


# ---------------------------------------------------------------------------
# validity gate + figures-side filtering
# ---------------------------------------------------------------------------


def test_cell_validity_floor_and_coherence() -> None:
    ok = fk._cell_validity({"accounting": {"frac_items_complete": 0.96}, "coherence_pass": True})
    assert ok["valid"] is True and ok["completeness_pass"] is True
    low = fk._cell_validity({"accounting": {"frac_items_complete": 0.90}, "coherence_pass": True})
    assert low["valid"] is False and low["completeness_pass"] is False
    none_fc = fk._cell_validity(
        {"accounting": {"frac_items_complete": None}, "coherence_pass": True}
    )
    assert none_fc["valid"] is False
    incoh = fk._cell_validity({"accounting": {"frac_items_complete": 1.0}, "coherence_pass": False})
    assert incoh["valid"] is False and incoh["coherence_pass"] is False


def test_figures_row_valid_semantics() -> None:
    assert figs._row_valid(None) is False
    assert figs._row_valid({"validity": {"valid": False}}) is False
    assert figs._row_valid({"validity": {"valid": True}}) is True
    # Legacy rows lacking the block stay plottable (treated valid).
    assert figs._row_valid({"delta_score": 1.0}) is True


def test_lattice_blocks_exclude_not_computable_variants() -> None:
    good = {"verdict": "Ambiguous", "R": {"point": 0.5, "lo": 0.1, "hi": 0.9}}
    lat = {
        "lattice": {
            "a": {"verdict": "not-computable pending remediation", "invalid_arms": ["x"]},
            "b": {"verdict": "not-computable", "note": "core arm missing"},
            "c": good,
        }
    }
    assert figs._lattice_blocks(lat) == [good]


# ---------------------------------------------------------------------------
# r3 BLOCKER firstk-pilot-pack-reachability: phase_judge uploads the pilot
# pack on EVERY pilot-section exit (mocked seams; no git/HF/API)
# ---------------------------------------------------------------------------


def _judge_args(tmp_path: Path, **over) -> argparse.Namespace:
    ns = argparse.Namespace(
        out_root=str(tmp_path / "out"),
        behaviors=["evil"],
        smoke=False,
        force=False,
        pilot=False,
        q_steer=20,
        draws=6,
        seed_base=42,
        waive_judge_parse_fail_arms=[],
    )
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _wire_phase_judge(monkeypatch, tmp_path: Path, cells: list[dict]) -> Path:
    """Mock every pre-pilot seam of ``phase_judge`` so its pilot-section
    wiring (pack upload on every exit) runs hermetically."""
    import explore_persona_space.experiments.issue_1739.judging as judging

    monkeypatch.setenv("EPM_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setattr(fk, "_ensure_git_inputs", lambda: None)
    monkeypatch.setattr(fk.i2254, "_load_operating_points", lambda root: {"ops": True})
    monkeypatch.setattr(fk, "resolve_operating_points", lambda ops, b: {"resolved": True})
    monkeypatch.setattr(fk, "build_cells", lambda a, r, b: list(cells))
    monkeypatch.setattr(fk.i2254, "_load_rho", lambda root: ({}, {}))
    monkeypatch.setattr(fk, "_steer_regime_fp", lambda a, c, r: "fp-test")
    comp_root = tmp_path / "comp"
    comp_root.mkdir(parents=True, exist_ok=True)
    for c in cells:
        (comp_root / f"{fk._cell_id(c)}.json").write_text("{}")
    monkeypatch.setattr(fk, "_stage_round_completions", lambda rroot, expected_fp: comp_root)
    monkeypatch.setattr(fk, "_validate_gen_grid", lambda a, cr, e, p: None)
    monkeypatch.setattr(judging, "load_trait_rubric", lambda b: f"rubric-{b}")
    return comp_root


def test_phase_judge_pilot_gate_fail_uploads_pilot_pack(tmp_path, monkeypatch) -> None:
    args = _judge_args(tmp_path)
    _wire_phase_judge(monkeypatch, tmp_path, [_cell()])
    uploads: list[Path] = []
    monkeypatch.setattr(fk, "_upload_pilot_pack_firstk", lambda rroot: uploads.append(rroot))

    def _gate_fail(*a, **k):
        raise RuntimeError("pilot gate FAILED (test)")

    monkeypatch.setattr(fk, "_run_firstk_pilot", _gate_fail)
    with pytest.raises(RuntimeError, match="pilot gate FAILED"):
        fk.phase_judge(args)
    assert uploads == [fk.round_root(Path(args.out_root))]


def test_phase_judge_pilot_flag_return_uploads_pilot_pack_and_skips_wave(
    tmp_path, monkeypatch
) -> None:
    args = _judge_args(tmp_path, pilot=True)
    _wire_phase_judge(monkeypatch, tmp_path, [_cell()])
    uploads: list[Path] = []
    monkeypatch.setattr(fk, "_upload_pilot_pack_firstk", lambda rroot: uploads.append(rroot))
    monkeypatch.setattr(fk, "_run_firstk_pilot", lambda *a, **k: None)

    def _never(*a, **k):
        raise AssertionError("wave must not dispatch on --pilot")

    monkeypatch.setattr(fk, "_judge_firstk_cell", _never)
    monkeypatch.setattr(fk, "_upload_judge_outputs_firstk", _never)
    fk.phase_judge(args)
    assert uploads == [fk.round_root(Path(args.out_root))]


def test_phase_judge_pc_kill_uploads_pilot_pack(tmp_path, monkeypatch) -> None:
    cells = [_cell(), dict(_cell(), behavior="sycophancy")]
    args = _judge_args(tmp_path, behaviors=["evil", "sycophancy"])
    _wire_phase_judge(monkeypatch, tmp_path, cells)
    uploads: list[Path] = []
    monkeypatch.setattr(fk, "_upload_pilot_pack_firstk", lambda rroot: uploads.append(rroot))

    def _pilot_pc_fail(args_, rroot, comp_root_, b, rubric, n_draws):
        p = rroot / "judge" / "pilot" / f"{b}.pass.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"positive_control_early": {"delta_rb_minus_random": -1.0}}))

    monkeypatch.setattr(fk, "_run_firstk_pilot", _pilot_pc_fail)
    with pytest.raises(RuntimeError, match="positive-control EARLY kill"):
        fk.phase_judge(args)
    assert uploads == [fk.round_root(Path(args.out_root))]


def test_phase_judge_upload_error_never_masks_pilot_kill(tmp_path, monkeypatch) -> None:
    args = _judge_args(tmp_path)
    _wire_phase_judge(monkeypatch, tmp_path, [_cell()])
    attempted: list[Path] = []

    def _upload_boom(rroot):
        attempted.append(rroot)
        raise OSError("hub down (test)")

    monkeypatch.setattr(fk, "_upload_pilot_pack_firstk", _upload_boom)

    def _gate_fail(*a, **k):
        raise RuntimeError("pilot gate FAILED (test)")

    monkeypatch.setattr(fk, "_run_firstk_pilot", _gate_fail)
    # The ORIGINAL pilot kill propagates; the upload error is logged, never a mask.
    with pytest.raises(RuntimeError, match="pilot gate FAILED"):
        fk.phase_judge(args)
    assert attempted == [fk.round_root(Path(args.out_root))]


def test_phase_judge_upload_error_raises_on_healthy_pilot(tmp_path, monkeypatch) -> None:
    args = _judge_args(tmp_path, pilot=True)
    _wire_phase_judge(monkeypatch, tmp_path, [_cell()])
    monkeypatch.setattr(fk, "_run_firstk_pilot", lambda *a, **k: None)

    def _upload_boom(rroot):
        raise OSError("hub down (test)")

    monkeypatch.setattr(fk, "_upload_pilot_pack_firstk", _upload_boom)
    # No in-flight pilot error => the upload failure is REAL and must raise.
    with pytest.raises(OSError, match="hub down"):
        fk.phase_judge(args)


# ---------------------------------------------------------------------------
# r3: pilot-pack uploader body (real pack; HF boundary faked)
# ---------------------------------------------------------------------------


def test_upload_pilot_pack_packs_whole_tree_and_uploads(tmp_path, monkeypatch) -> None:
    rroot = tmp_path / "round"
    pilot = rroot / "judge" / "pilot"
    (pilot / "evil_raw").mkdir(parents=True)
    (pilot / "evil.pass.json").write_text('{"fingerprint": "x"}')
    (pilot / "evil_raw" / "judge_raw_pilot_00").write_text('{"raw": 1}')  # extensionless
    ups: list[tuple[Path, str]] = []
    monkeypatch.setattr(
        fk.i2254,
        "_upload_folder_to_hf",
        lambda dest, prefix, allow=None: ups.append((Path(dest), prefix)),
    )
    fk._upload_pilot_pack_firstk(rroot)
    assert len(ups) == 1
    dest, prefix = ups[0]
    assert prefix.endswith("/judge/pilot_pack")
    manifest = json.loads((dest / "pack_manifest.json").read_text())
    assert manifest["n_files"] == 2  # pattern='*' packs the extensionless raw too
    rows = [
        json.loads(ln)
        for shard in manifest["shards"]
        for ln in (dest / shard).read_text().splitlines()
        if ln.strip()
    ]
    assert {Path(r["path"]).name for r in rows} == {"evil.pass.json", "judge_raw_pilot_00"}


def test_upload_pilot_pack_noop_without_pilot_artifacts(tmp_path, monkeypatch) -> None:
    called: list[int] = []
    monkeypatch.setattr(fk.i2254, "_upload_folder_to_hf", lambda *a, **k: called.append(1))
    fk._upload_pilot_pack_firstk(tmp_path / "round")  # pilot dir absent
    assert not called


# ---------------------------------------------------------------------------
# r3: manifest-driven pack rehydration + regime_fp staging cross-checks
# ---------------------------------------------------------------------------


def _fake_remote(monkeypatch, remote: dict[str, bytes]) -> None:
    def _tree(prefix: str, *, recursive: bool = False) -> list:
        return [SimpleNamespace(path=p) for p in sorted(remote) if p.startswith(prefix + "/")]

    def _stage(path_in_repo: str, target: Path) -> None:
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(remote[path_in_repo])

    monkeypatch.setattr(fk, "_hub_tree", _tree)
    monkeypatch.setattr(fk, "_hub_stage", _stage)


def _pack_row(name: str, doc: dict) -> str:
    return json.dumps({"path": f"stage/{name}", "doc": doc})


def test_stage_manifest_driven_ignores_stale_tail(tmp_path, monkeypatch) -> None:
    pp = f"{fk._round_hf_prefix()}/raw_completions/steer_pack"
    fresh_a = {"cell_id": "cellA", "regime_fp": "fpA", "vintage": "fresh"}
    fresh_b = {"cell_id": "cellB", "regime_fp": "fpB", "vintage": "fresh"}
    stale_a = {"cell_id": "cellA", "regime_fp": "stale", "vintage": "stale"}
    manifest = {"group": "g", "n_files": 2, "shards": ["g.shard00.jsonl"]}
    _fake_remote(
        monkeypatch,
        {
            f"{pp}/shard0/pack_manifest.json": json.dumps(manifest).encode(),
            f"{pp}/shard0/g.shard00.jsonl": (
                _pack_row("cellA.json", fresh_a) + "\n" + _pack_row("cellB.json", fresh_b) + "\n"
            ).encode(),
            # Stale tail from a shrinking repack: NOT in the manifest -> ignored.
            f"{pp}/shard0/g.shard01.jsonl": (_pack_row("cellA.json", stale_a) + "\n").encode(),
        },
    )
    out = fk._stage_round_completions(tmp_path / "round", {"cellA": "fpA", "cellB": "fpB"})
    assert json.loads((out / "cellA.json").read_text())["vintage"] == "fresh"
    assert json.loads((out / "cellB.json").read_text())["regime_fp"] == "fpB"


def test_stage_refuses_duplicate_cell_paths_across_packs(tmp_path, monkeypatch) -> None:
    pp = f"{fk._round_hf_prefix()}/raw_completions/steer_pack"
    doc = {"cell_id": "cellA", "regime_fp": "fpA"}
    m = {"group": "g", "n_files": 1, "shards": ["g.shard00.jsonl"]}
    _fake_remote(
        monkeypatch,
        {
            f"{pp}/shard0/pack_manifest.json": json.dumps(m).encode(),
            f"{pp}/shard0/g.shard00.jsonl": (_pack_row("cellA.json", doc) + "\n").encode(),
            f"{pp}/shard1/pack_manifest.json": json.dumps(m).encode(),
            f"{pp}/shard1/g.shard00.jsonl": (_pack_row("cellA.json", doc) + "\n").encode(),
        },
    )
    with pytest.raises(RuntimeError, match="duplicate cell record"):
        fk._stage_round_completions(tmp_path / "round", {"cellA": "fpA"})


def test_stage_refuses_unmanifested_shards(tmp_path, monkeypatch) -> None:
    pp = f"{fk._round_hf_prefix()}/raw_completions/steer_pack"
    doc = {"cell_id": "cellA", "regime_fp": "fpA"}
    _fake_remote(
        monkeypatch,
        {f"{pp}/shard0/g.shard00.jsonl": (_pack_row("cellA.json", doc) + "\n").encode()},
    )
    with pytest.raises(RuntimeError, match="pack_manifest"):
        fk._stage_round_completions(tmp_path / "round", {"cellA": "fpA"})


def test_stage_refuses_manifest_row_count_mismatch(tmp_path, monkeypatch) -> None:
    pp = f"{fk._round_hf_prefix()}/raw_completions/steer_pack"
    doc = {"cell_id": "cellA", "regime_fp": "fpA"}
    m = {"group": "g", "n_files": 3, "shards": ["g.shard00.jsonl"]}  # declares 3, ships 1
    _fake_remote(
        monkeypatch,
        {
            f"{pp}/shard0/pack_manifest.json": json.dumps(m).encode(),
            f"{pp}/shard0/g.shard00.jsonl": (_pack_row("cellA.json", doc) + "\n").encode(),
        },
    )
    with pytest.raises(RuntimeError, match="n_files"):
        fk._stage_round_completions(tmp_path / "round", {"cellA": "fpA"})


def test_stage_refuses_manifest_named_shard_missing_remotely(tmp_path, monkeypatch) -> None:
    pp = f"{fk._round_hf_prefix()}/raw_completions/steer_pack"
    m = {"group": "g", "n_files": 1, "shards": ["g.shard00.jsonl"]}
    _fake_remote(monkeypatch, {f"{pp}/shard0/pack_manifest.json": json.dumps(m).encode()})
    with pytest.raises(RuntimeError, match="absent from the remote listing"):
        fk._stage_round_completions(tmp_path / "round", {"cellA": "fpA"})


def test_stage_localfirst_validates_regime_fp(tmp_path, monkeypatch) -> None:
    _fake_remote(monkeypatch, {})  # never reached on the local-first branch
    rroot = tmp_path / "round"
    comp_root = rroot / "steer" / "raw_completions"
    comp_root.mkdir(parents=True)
    (comp_root / "cellA.json").write_text(json.dumps({"cell_id": "cellA", "regime_fp": "fpA"}))
    assert fk._stage_round_completions(rroot, {"cellA": "fpA"}) == comp_root
    with pytest.raises(RuntimeError, match="regime_fp"):
        fk._stage_round_completions(rroot, {"cellA": "OTHER"})


def test_stage_raises_when_nothing_anywhere(tmp_path, monkeypatch) -> None:
    _fake_remote(monkeypatch, {})
    with pytest.raises(RuntimeError, match="no steer completions"):
        fk._stage_round_completions(tmp_path / "round", {"cellA": "fpA"})


# ---------------------------------------------------------------------------
# r3: reduce-side judged-vintage gate (r2 Codex C3)
# ---------------------------------------------------------------------------


def test_assert_judged_vintage_pass_and_refusals(tmp_path) -> None:
    gp = tmp_path / "cellA.json"
    gp.write_bytes(b'{"x": 1}')
    sha = hashlib.sha256(gp.read_bytes()).hexdigest()[:12]
    j = {"cell_id": "cellA", "gen_sha": sha, "judge_fp": "jfp"}
    fk._assert_judged_vintage(j, gp, "jfp")  # matching vintage passes
    with pytest.raises(RuntimeError, match="gen_sha"):
        fk._assert_judged_vintage({**j, "gen_sha": "deadbeef0000"}, gp, "jfp")
    with pytest.raises(RuntimeError, match="judge_fp"):
        fk._assert_judged_vintage(j, gp, "other-instrument")


def test_judge_instrument_fp_keys_on_draws() -> None:
    assert fk._judge_instrument_fp("rubric", 5) != fk._judge_instrument_fp("rubric", 2)
    assert fk._judge_instrument_fp("rubric", 5) == fk._judge_instrument_fp("rubric", 5)


# ---------------------------------------------------------------------------
# r3: pilot-PASS input fingerprint (regen'd gen cells invalidate a prior PASS)
# ---------------------------------------------------------------------------


def test_pilot_gen_hash_changes_on_regenerated_input(tmp_path) -> None:
    a = tmp_path / "evil__a.json"
    b = tmp_path / "evil__b.json"
    a.write_text('{"v": 1}')
    b.write_text('{"v": 2}')
    h1 = fk._pilot_gen_hash([a, b])
    assert fk._pilot_gen_hash([a, b]) == h1  # deterministic on identical bytes
    a.write_text('{"v": 999}')  # a regen'd source cell
    assert fk._pilot_gen_hash([a, b]) != h1


# ---------------------------------------------------------------------------
# r3: empty-draw regen-once escape (validator-wedge)
# ---------------------------------------------------------------------------


def test_empty_draw_slots_finds_empty_and_nonstr() -> None:
    cell = _cell()
    rec = _gen_rec(cell, fk._cell_id(cell))
    assert fk._empty_draw_slots(rec) == []
    rec["seeds"]["42"]["completions"][1][0] = ""
    rec["seeds"]["42"]["completions"][2][1] = None
    assert fk._empty_draw_slots(rec) == [("42", 1, 0), ("42", 2, 1)]


def test_regen_empty_draw_cell_noop_when_clean() -> None:
    cell = _cell()
    rec = _gen_rec(cell, fk._cell_id(cell))

    def _never(**k):
        raise AssertionError("clean cell must not regen")

    assert fk._regen_empty_draw_cell(_never, "cid", rec, seed_base=42) is rec


def test_regen_empty_draw_cell_retries_once_at_shifted_seed() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    rec = _gen_rec(cell, cid)
    rec["regen"] = {"initial_cap_hit_fraction": 0.5}
    rec["seeds"]["42"]["completions"][0][1] = ""
    seen: list[int] = []

    def _gen(seed_base: int) -> dict:
        seen.append(seed_base)
        return _gen_rec(cell, cid)  # clean retry

    out = fk._regen_empty_draw_cell(_gen, cid, rec, seed_base=42)
    assert seen == [42 + fk.EMPTY_DRAW_SEED_SHIFT]
    assert out["empty_draw_regen"]["n_empty_initial"] == 1
    assert out["empty_draw_regen"]["seed_base_retry"] == 42 + fk.EMPTY_DRAW_SEED_SHIFT
    assert out["regen"] == {"initial_cap_hit_fraction": 0.5}  # cap-regen audit preserved


def test_regen_empty_draw_cell_fails_loud_when_empty_persists() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    rec = _gen_rec(cell, cid)
    rec["seeds"]["42"]["completions"][0][1] = ""

    def _gen(seed_base: int) -> dict:
        bad = _gen_rec(cell, cid)
        bad["seeds"]["42"]["completions"][0][1] = ""  # still empty at the shifted seed
        return bad

    with pytest.raises(RuntimeError, match="deterministic per-draw seeds"):
        fk._regen_empty_draw_cell(_gen, cid, rec, seed_base=42)


def test_validate_gen_record_empty_draw_message_names_the_wedge() -> None:
    cell = _cell()
    cid = fk._cell_id(cell)
    rec = _gen_rec(cell, cid)
    rec["seeds"]["42"]["completions"][1][0] = ""
    with pytest.raises(AssertionError, match="min_new_tokens"):
        fk._validate_gen_record(rec, Path(f"{cid}.json"), n_q=3, n_draws=2)
