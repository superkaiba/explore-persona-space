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


# ---------------------------------------------------------------------------
# round 4 — rule-28 sync re-issue (shared helper + pilot post-reissue kill)
# ---------------------------------------------------------------------------


def _jres(
    scores: dict[str, list[float]],
    refusals: dict[str, int] | None = None,
    *,
    total: int,
    transport: int = 0,
) -> SimpleNamespace:
    """JudgeResult stand-in with exactly the fields the r4 reissue path reads."""
    refusals = refusals or {}
    return SimpleNamespace(
        per_item_scores=scores,
        per_item_api_refusals=refusals,
        n_total_draws=total,
        n_transport_lost_draws=transport,
        n_api_refusal_draws=sum(refusals.values()),
    )


def _fake_judge_items_graded(script):
    """Signature-mirroring ``judge_items_graded`` fake driven by
    ``script(items, threshold_base, n_draws) -> result``; records every call."""
    calls: list[dict] = []

    def fake(
        items,
        eval_prompt,
        *,
        cache_dir,
        save_raw,
        n_draws=3,
        temperature=1.0,
        max_tokens=2048,
        judge_model="fake-judge",
        dry_run=False,
        threshold_base=None,
    ):
        calls.append(
            {
                "items": [it[0] for it in items],
                "cache_dir": Path(cache_dir),
                "save_raw": Path(save_raw),
                "n_draws": n_draws,
                "tb": threshold_base,
            }
        )
        return script(items, threshold_base, n_draws)

    return fake, calls


def test_rule28_helper_merges_sync_scores_and_never_redispatches_batch_successes(
    tmp_path, monkeypatch
) -> None:
    """(a) batch-then-sync MERGE: batch survivors untouched (never
    re-dispatched), censored draws re-drawn sync-side in per-k groups with
    n_draws=k against FRESH ``_syncfix_k{k}`` sibling cache dirs (rule 24(ii):
    a same-cache replay would duplicate a surviving sibling draw)."""
    from explore_persona_space.experiments.issue_1739 import judging

    def script(items, tb, n_draws):
        if tb == fk.JUDGE_THRESHOLD_BASE_BATCH:
            # pass 1 (batch): a fully scored; b partially censored (1 valid +
            # 2 api-refused); c fully censored (0 valid + 3 api-refused)
            return _jres({"a": [80.0, 70.0, 90.0], "b": [50.0], "c": []}, {"b": 2, "c": 3}, total=9)
        assert tb == fk.SYNC_FORCE_THRESHOLD_BASE  # the sync lever, never batch
        return _jres({it[0]: [42.0] * n_draws for it in items}, total=n_draws * len(items))

    fake, calls = _fake_judge_items_graded(script)
    monkeypatch.setattr(judging, "judge_items_graded", fake)
    items = [("a", "q", "ans"), ("b", "q", "ans"), ("c", "q", "ans")]
    result, merged, reissue = fk._judge_graded_with_refusal_reissue(
        items,
        "RUBRIC",
        cache_dir=tmp_path / "cache" / "cell1",
        save_raw=tmp_path / "raw" / "cell1",
        n_draws=3,
    )
    assert merged == {
        "a": [80.0, 70.0, 90.0],
        "b": [50.0, 42.0, 42.0],
        "c": [42.0, 42.0, 42.0],
    }
    assert result.n_total_draws == 9  # pass-1 JudgeResult returned for accounting
    sync = [c for c in calls if c["tb"] == fk.SYNC_FORCE_THRESHOLD_BASE]
    assert [c["items"] for c in sync] == [["b"], ["c"]]  # k=2 group then k=3 group
    assert [c["n_draws"] for c in sync] == [2, 3]  # exactly the censored draw counts
    assert all("a" not in c["items"] for c in sync)  # batch successes never re-dispatched
    assert sync[0]["cache_dir"] == tmp_path / "cache" / "cell1_syncfix_k2"
    assert sync[1]["save_raw"] == tmp_path / "raw" / "cell1_syncfix_k3"
    assert reissue["n_items_reissued"] == 2
    assert reissue["n_draws_reissued"] == 5
    assert reissue["n_scored"] == 5
    assert reissue["n_api_refusal_residual"] == 0


def test_rule28_helper_no_refusals_is_a_single_batch_pass(tmp_path, monkeypatch) -> None:
    """Zero censored draws => exactly one (batch) judge call, no reissue."""
    from explore_persona_space.experiments.issue_1739 import judging

    fake, calls = _fake_judge_items_graded(
        lambda items, tb, n_draws: _jres({"a": [80.0, 70.0]}, total=2)
    )
    monkeypatch.setattr(judging, "judge_items_graded", fake)
    _result, merged, reissue = fk._judge_graded_with_refusal_reissue(
        [("a", "q", "ans")],
        "RUBRIC",
        cache_dir=tmp_path / "cache" / "cell1",
        save_raw=tmp_path / "raw" / "cell1",
        n_draws=2,
    )
    assert reissue is None and merged == {"a": [80.0, 70.0]}
    assert len(calls) == 1 and calls[0]["tb"] == fk.JUDGE_THRESHOLD_BASE_BATCH


def test_rule28_helper_fails_loud_on_uncensorable_residual(tmp_path, monkeypatch) -> None:
    """(c) rows still api-refused AFTER the bounded sync pass, above the
    plan-§6 bound, halt LOUD — never a silent drop (rule 28: the censoring is
    outcome-correlated)."""
    from explore_persona_space.experiments.issue_1739 import judging

    def script(items, tb, n_draws):
        if tb == fk.JUDGE_THRESHOLD_BASE_BATCH:
            return _jres({"a": [], "b": [70.0, 60.0, 80.0]}, {"a": 3}, total=6)
        # sync pass: every reissued draw refuses AGAIN (residual 3/6 = 0.5)
        return _jres(
            {it[0]: [] for it in items},
            {it[0]: n_draws for it in items},
            total=n_draws * len(items),
        )

    fake, _calls = _fake_judge_items_graded(script)
    monkeypatch.setattr(judging, "judge_items_graded", fake)
    with pytest.raises(RuntimeError, match="AFTER the bounded rule-28 sync re-issue"):
        fk._judge_graded_with_refusal_reissue(
            [("a", "q", "ans"), ("b", "q", "ans")],
            "RUBRIC",
            cache_dir=tmp_path / "c" / "x",
            save_raw=tmp_path / "r" / "x",
            n_draws=3,
        )


def test_rule28_helper_tolerates_sub_threshold_residual_and_surfaces_it(
    tmp_path, monkeypatch
) -> None:
    """A sub-threshold uncensorable tail (rule 28's 'genuinely uncensorable
    row') is kept + surfaced in the reissue accounting, never dropped and
    never a spurious halt (residual 1/60 = 0.017 < 0.10)."""
    from explore_persona_space.experiments.issue_1739 import judging

    def script(items, tb, n_draws):
        if tb == fk.JUDGE_THRESHOLD_BASE_BATCH:
            return _jres({"a": [50.0, 60.0], "b": [70.0, 75.0, 65.0]}, {"a": 1}, total=60)
        return _jres({"a": []}, {"a": n_draws}, total=n_draws * len(items))

    fake, _calls = _fake_judge_items_graded(script)
    monkeypatch.setattr(judging, "judge_items_graded", fake)
    _result, merged, reissue = fk._judge_graded_with_refusal_reissue(
        [("a", "q", "ans"), ("b", "q", "ans")],
        "RUBRIC",
        cache_dir=tmp_path / "c" / "x",
        save_raw=tmp_path / "r" / "x",
        n_draws=3,
    )
    assert merged["a"] == [50.0, 60.0]  # surviving batch draws kept
    assert reissue["n_api_refusal_residual"] == 1  # surfaced, not silently dropped


def _arm_stats(n_refused: int, n_draws: int = 55, n_transport: int = 0) -> SimpleNamespace:
    return SimpleNamespace(n_api_refusal=n_refused, n_draws=n_draws, n_transport_lost=n_transport)


def _pilot_env(monkeypatch, *, post_refused: dict[str, int], post_passed: bool = True):
    """Wire the two boundary fakes for the pilot remediation flow: sync
    ``judge_items_graded`` always succeeds; the gate re-run reports
    ``post_refused`` refusals per arm off the warmed cache."""
    from explore_persona_space.eval import judge_pilot as jp
    from explore_persona_space.experiments.issue_1739 import judging

    def script(items, tb, n_draws):
        assert tb == fk.SYNC_FORCE_THRESHOLD_BASE  # the pilot reissue is sync-only
        return _jres({it[0]: [10.0] * n_draws for it in items}, total=n_draws * len(items))

    fake_judge, judge_calls = _fake_judge_items_graded(script)
    monkeypatch.setattr(judging, "judge_items_graded", fake_judge)
    gate_calls: list[dict] = []

    def fake_gate(arms, rubric, **kw):
        gate_calls.append(kw)
        return SimpleNamespace(
            passed=post_passed,
            verdict="PASS" if post_passed else "FAIL",
            arms={a: _arm_stats(post_refused.get(a, 0)) for a in arms},
        )

    monkeypatch.setattr(jp, "judge_pilot_gate", fake_gate)
    return judge_calls, gate_calls


def _pilot_arms() -> dict[str, list[tuple[str, str, str]]]:
    return {
        "rb_allans": [(f"rb-{i:03d}", "q", "a") for i in range(11)],
        "clean_opening": [(f"cl-{i:03d}", "q", "a") for i in range(11)],
    }


def test_pilot_rule28_kill_applies_to_post_reissue_rate_pass(tmp_path, monkeypatch) -> None:
    """(b) batch 0.273 -> sync-recovered 0.000: the plan-§6 kill reads the
    POST-reissue rate off the gate re-run and the pilot PROCEEDS; only the
    flagged arm re-issues, against its SAME arm cache, one bounded re-run."""
    judge_calls, gate_calls = _pilot_env(monkeypatch, post_refused={})
    args = SimpleNamespace(smoke=False, waive_judge_parse_fail_arms=[])
    report = SimpleNamespace(
        passed=True,
        verdict="PASS",
        arms={"rb_allans": _arm_stats(15), "clean_opening": _arm_stats(2)},
    )
    raw, post, meta, report_post = fk._pilot_refusal_remediation_and_kill(
        args, tmp_path, "evil", "RUBRIC", 5, _pilot_arms(), 11, report
    )
    assert raw["rb_allans"] == pytest.approx(15 / 55)  # 0.273 raw batch rate
    assert post == {"rb_allans": 0.0, "clean_opening": 0.0}
    assert list(meta) == ["rb_allans"]  # only the flagged arm re-issued
    assert meta["rb_allans"]["rate_batch"] == pytest.approx(15 / 55)
    assert len(gate_calls) == 1  # ONE bounded gate re-run
    assert judge_calls and all(c["tb"] == fk.SYNC_FORCE_THRESHOLD_BASE for c in judge_calls)
    # SAME arm cache (the api-refusal cache-miss invariant targets the
    # censored rows); DISTINCT raw + report paths preserve the batch evidence.
    assert judge_calls[0]["cache_dir"] == tmp_path / "evil_cache" / "rb_allans"
    assert judge_calls[0]["save_raw"] == tmp_path / "evil_raw" / "syncreissue_rb_allans.json"
    assert gate_calls[0]["save_raw_dir"] == tmp_path / "evil_raw_postreissue"
    assert gate_calls[0]["report_path"] == tmp_path / "evil.report_postreissue.json"
    assert gate_calls[0]["threshold_base"] == fk.JUDGE_THRESHOLD_BASE_BATCH
    assert report_post.passed


def test_pilot_rule28_kill_fires_when_post_reissue_rate_still_high(tmp_path, monkeypatch) -> None:
    """(b) batch 0.273 -> STILL 0.273 after the bounded sync re-issue: the
    genuine plan-§6 kill fires (halt-and-report path preserved)."""
    _pilot_env(monkeypatch, post_refused={"rb_allans": 15})
    args = SimpleNamespace(smoke=False, waive_judge_parse_fail_arms=[])
    report = SimpleNamespace(
        passed=True,
        verdict="PASS",
        arms={"rb_allans": _arm_stats(15), "clean_opening": _arm_stats(0)},
    )
    with pytest.raises(RuntimeError, match="AFTER the bounded rule-28 sync re-issue"):
        fk._pilot_refusal_remediation_and_kill(
            args, tmp_path, "evil", "RUBRIC", 5, _pilot_arms(), 11, report
        )


def test_pilot_rule28_no_flagged_arm_is_a_no_op(tmp_path, monkeypatch) -> None:
    """No arm at/over 0.10 => zero reissue spend, no gate re-run, raw rates
    returned unchanged (the untouched common path)."""
    judge_calls, gate_calls = _pilot_env(monkeypatch, post_refused={})
    args = SimpleNamespace(smoke=False, waive_judge_parse_fail_arms=[])
    report = SimpleNamespace(
        passed=True,
        verdict="PASS",
        arms={"rb_allans": _arm_stats(2), "clean_opening": _arm_stats(0)},
    )
    raw, post, meta, report_post = fk._pilot_refusal_remediation_and_kill(
        args, tmp_path, "evil", "RUBRIC", 5, _pilot_arms(), 11, report
    )
    assert raw["rb_allans"] == pytest.approx(2 / 55)
    assert (post, meta, report_post) == (None, None, None)
    assert not judge_calls and not gate_calls


def test_pilot_rule28_reissue_gate_rerun_fail_raises(tmp_path, monkeypatch) -> None:
    """A post-reissue gate re-run that FAILs (parse-fail/truncation floors off
    the warmed cache) halts loud rather than proceeding to the wave."""
    _pilot_env(monkeypatch, post_refused={}, post_passed=False)
    args = SimpleNamespace(smoke=False, waive_judge_parse_fail_arms=[])
    report = SimpleNamespace(
        passed=True,
        verdict="PASS",
        arms={"rb_allans": _arm_stats(15), "clean_opening": _arm_stats(0)},
    )
    with pytest.raises(RuntimeError, match="post-reissue gate re-run"):
        fk._pilot_refusal_remediation_and_kill(
            args, tmp_path, "evil", "RUBRIC", 5, _pilot_arms(), 11, report
        )
