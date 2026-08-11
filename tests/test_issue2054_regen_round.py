"""Regression pins for the #2054 `coordinated-common-set-regen` round
(plan v12 §4): fold-map extension invariants, wave pending-set arithmetic,
the ARMED fits fleet-wall fence, the gate-1 pair census, and the gate-2 KS
fire logic. Synthetic fixtures only — no network, no GPU, no repo-resident
eval artifacts (sparse-worktree safe)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO / "scripts"), str(_REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2054_regen_waves as rw  # noqa: E402


# ---------------------------------------------------------------------------
# Fold-map extension (plan §4 R0 step 3 / fact-check A38)
# ---------------------------------------------------------------------------
def _synthetic_ref(tmp_path: Path, n: int) -> Path:
    import issue2054_phase_a as pa

    ids = [f"mt_ref{i:08x}" for i in range(n)]
    fold_of = pa._conv_grouped_folds(ids, k=5, seed=137)
    p = tmp_path / "ref.json"
    p.write_text(
        json.dumps({"artifact": "shared_fold_map", "k": 5, "seed": 137, "fold_of": fold_of})
    )
    return p


def _draw_args(tmp_path: Path, ref: Path) -> argparse.Namespace:
    return argparse.Namespace(
        fold_map_ref=str(ref),
        fold_map_out=str(tmp_path / "extended.json"),
        seed=137,
    )


def test_fold_map_extension_preserves_existing_assignments(tmp_path):
    ref = _synthetic_ref(tmp_path, rw.FOLD_MAP_REF_MIN_KEYS + 1)
    t_ids = [f"mt_new{i:04d}" for i in range(25)]
    out = rw._extend_fold_map(_draw_args(tmp_path, ref), tmp_path, t_ids)
    ext = json.loads(out.read_text())
    ref_fold = json.loads(ref.read_text())["fold_of"]
    assert ext["n_reference"] == len(ref_fold)
    assert ext["n_new"] == 25
    for cid, fold in list(ref_fold.items())[:500]:
        assert ext["fold_of"][cid] == fold  # old assignments preserved
    assert all(cid in ext["fold_of"] for cid in t_ids)


def test_fold_map_extension_refuses_stale_main_copy(tmp_path):
    ref = _synthetic_ref(tmp_path, 1_761)  # the stale repo-root main copy's size
    with pytest.raises(RuntimeError, match="STALE repo-root main copy"):
        rw._extend_fold_map(_draw_args(tmp_path, ref), tmp_path, ["mt_new0001"])


# ---------------------------------------------------------------------------
# Wave pending-set arithmetic (plan §4 R1: <=3 attempts; wave 4 caps at 4)
# ---------------------------------------------------------------------------
def _state(admitted: dict, attempts: dict, waves: list[dict]) -> dict:
    return {"admitted": admitted, "attempts": attempts, "waves_done": waves}


def test_pending_excludes_admitted_and_caps_attempts():
    t_ids = [f"c{i}" for i in range(6)]
    admitted = {v: [] for v in rw.CHAR_VARIANTS}
    attempts = {v: {} for v in rw.CHAR_VARIANTS}
    admitted["char_helios"] = ["c0"]
    attempts["char_helios"] = {"c0": 1, "c1": 3, "c2": 2}
    pending = rw._pending_for_wave(_state(admitted, attempts, []), t_ids, wave=2)
    # admitted c0 out; c1 at the 3-attempt cap out; c2 (2 attempts) still in.
    assert pending["char_helios"] == ["c2", "c3", "c4", "c5"]
    assert pending["char_wren"] == t_ids


def test_contingency_wave_prioritizes_fewest_missing_and_allows_4th_attempt():
    t_ids = [f"c{i}" for i in range(4)]
    # c0: fully admitted (in S). c1: missing ONE character. c2/c3: missing all.
    admitted = {v: ["c0"] for v in rw.CHAR_VARIANTS}
    for v in rw.CHAR_VARIANTS[1:]:
        admitted[v] = sorted(admitted[v] + ["c1"])
    attempts = {v: {c: 3 for c in t_ids} for v in rw.CHAR_VARIANTS}  # 3 used; 4th allowed
    waves = [{"wave": 2, "n_attempted": 100, "n_admitted_new": 50, "survivors": 1}]
    pending = rw._pending_for_wave(_state(admitted, attempts, waves), t_ids, wave=4)
    # gap = 9000 - |S| with rate 0.5 -> every non-S conversation eligible;
    # the missing-one conversation c1 must be selected for exactly its
    # missing character, before the missing-all conversations.
    assert "c1" in pending["char_helios"]
    assert all("c1" not in pending[v] for v in rw.CHAR_VARIANTS[1:])


def test_measured_retry_rate_pools_waves_2_plus_first():
    waves = [
        {"wave": 1, "n_attempted": 100, "n_admitted_new": 80, "survivors": 0},
        {"wave": 2, "n_attempted": 50, "n_admitted_new": 20, "survivors": 0},
    ]
    assert rw._measured_retry_rate({"waves_done": waves}) == pytest.approx(0.4)
    assert rw._measured_retry_rate({"waves_done": waves[:1]}) == pytest.approx(0.8)
    assert rw._measured_retry_rate({"waves_done": []}) is None


# ---------------------------------------------------------------------------
# Fits fleet-wall fence (plan §4 item 6 — previously WARN-only; now ARMED)
# ---------------------------------------------------------------------------
def test_fits_pilot_gate_prior_report_enforces_armed_fence(tmp_path):
    import issue2054_fits as fits
    from issue2054_pilot import FleetWallExceeded

    report = {
        "n_null_draws": 100,
        "bootstrap_draws": 200,
        "arm": "context",
        "seed": 137,
        "wall_seconds": 3600.0,  # 1 unit-fold hour -> any fleet blows a tiny budget
    }
    (tmp_path / "pilot_gate_report.json").write_text(json.dumps(report))
    args = argparse.Namespace(
        arms=["context"],
        seed=137,
        n_null_draws=100,
        bootstrap_draws=200,
        overwrite=False,
        max_fleet_wall_hours=0.001,
    )
    with pytest.raises(FleetWallExceeded):
        fits._run_fits_pilot_gate(
            activations_by_cell={("v", "inserted", "chat", "m"): object()},
            groups={},
            group_restrict={},
            fold_map={"k": 5},
            args=args,
            output_dir=tmp_path,
        )


# ---------------------------------------------------------------------------
# Gate-1 pair census (plan §7 gate 1; fail-open-(1) structural fix)
# ---------------------------------------------------------------------------
def test_gate1_pair_census_matches_committed_counts(tmp_path):
    import issue2054_gate1_intersections as g1

    survivors = tmp_path / "survivors.json"
    survivors.write_text(json.dumps({"survivor_conv_ids": [f"mt_{i:04d}" for i in range(10)]}))
    composition = tmp_path / "composition.json"
    composition.write_text(json.dumps({"class_prose": {"n": 96}, "class_twobytwo": {"n": 208}}))
    chat_census = tmp_path / "chat_census.json"
    chat_census.write_text(json.dumps({"n_context_arm": 32}))
    args = argparse.Namespace(
        survivors=str(survivors),
        composition=str(composition),
        chat_census=str(chat_census),
        report_out=str(tmp_path / "gate1_report.json"),
    )
    rc = g1.run_gate1(args)
    assert rc == g1.EXIT_ABORT  # |S| = 10 < 4,480: the designed abort
    report = json.loads((tmp_path / "gate1_report.json").read_text())
    assert report["verdict"] == "ABORT"
    assert report["affected_pairs"] == {"cross_character": 120, "twobytwo": 224, "total": 344}
    for check in report["census_checks"].values():
        assert check["enumerated"] == check["expected"]


def test_gate1_census_mismatch_fails_loud(tmp_path):
    import issue2054_gate1_intersections as g1

    survivors = tmp_path / "survivors.json"
    survivors.write_text(json.dumps({"survivor_conv_ids": ["mt_0001"]}))
    composition = tmp_path / "composition.json"
    composition.write_text(json.dumps({"class_prose": {"n": 95}, "class_twobytwo": {"n": 208}}))
    chat_census = tmp_path / "chat_census.json"
    chat_census.write_text(json.dumps({"n_context_arm": 32}))
    args = argparse.Namespace(
        survivors=str(survivors),
        composition=str(composition),
        chat_census=str(chat_census),
        report_out=str(tmp_path / "gate1_report.json"),
    )
    with pytest.raises(RuntimeError, match="pair-census mismatch"):
        g1.run_gate1(args)


# ---------------------------------------------------------------------------
# Gate-2 KS fire logic (v8 kill-gate-5 constants)
# ---------------------------------------------------------------------------
def test_gate2_pair_stats_fire_semantics():
    import issue2054_answer_lengths as al

    ids = [f"c{i}" for i in range(50)]
    short = {c: 10 + (i % 3) for i, c in enumerate(ids)}
    long = {c: 80 + (i % 7) for i, c in enumerate(ids)}
    fired = al._pair_stats(short, long)
    assert fired["fired"] is True  # KS 1.0, ratio ~0.13 (outside [0.25, 4.0])
    same = al._pair_stats(short, dict(short))
    assert same["fired"] is False
    assert same["n_matched"] == 50


def test_args_attrs_completeness_on_round_scripts():
    for mod in (
        "issue2054_regen_waves.py",
        "issue2054_gate1_intersections.py",
        "issue2054_answer_lengths.py",
    ):
        rw.assert_args_attrs_defined(_REPO / "scripts" / mod)


# ---------------------------------------------------------------------------
# HF-prefix rebind seam (plan §4 item 4 — the upload-prefix-clobber remedy)
# ---------------------------------------------------------------------------
def test_hf_prefix_rebind_rewires_upload_and_answer_source():
    import issue2054_phase_d as pd

    old_prefix, old_src = pd.TASK_PREFIX, pd.ANSWER_SOURCE
    try:
        pd._apply_hf_prefix("issue2054_lattice/common_regen")
        assert pd.TASK_PREFIX == "issue2054_lattice/common_regen"
        assert pd.ANSWER_SOURCE == "issue2054_lattice/common_regen/on_policy"
        assert pd._pool_path_in_repo("char_dana_op", "attrib_quoted").startswith(
            "issue2054_lattice/common_regen/on_policy/"
        )
    finally:
        pd._apply_hf_prefix(old_prefix)
        assert old_src == pd.ANSWER_SOURCE


# ---------------------------------------------------------------------------
# Production-body tests for the network-bound wave-driver stages (the
# external Hub boundary is the ONLY fake; signature-conformant by
# construction — code-style.md § One production-body test).
# ---------------------------------------------------------------------------
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_assist_merge_body_reaches_full_s_coverage(tmp_path, monkeypatch):
    import unittest.mock as um

    import huggingface_hub
    import issue2054_forms as forms

    out = tmp_path / "regen"
    s_ids = [f"mt_{i:04d}" for i in range(8)]
    (out / "scaffolds").mkdir(parents=True)
    (out / "scaffolds" / "survivor_set.json").write_text(json.dumps({"survivor_conv_ids": s_ids}))
    a = rw.ASSISTANT_VARIANT
    model = "qwen2.5-7b-instruct"
    fname = forms.phase_output_name("on_policy", a, "chat")
    # Delta rows (this round's generations): the last 3 survivors.
    _write_jsonl(
        out / "on_policy" / model / a / fname,
        [{"conv_id": c, "answer": "delta"} for c in s_ids[5:]],
    )
    _write_jsonl(
        out / "on_policy" / model / a / forms.phase_output_name("on_policy", a, "bare_text"),
        [{"conv_id": c, "answer": "delta"} for c in s_ids[5:]],
    )
    # Parent realized rows: the first 5 survivors + 2 non-survivors.
    parent = tmp_path / "parent.jsonl"
    _write_jsonl(
        parent,
        [{"conv_id": c, "answer": "parent"} for c in [*s_ids[:5], "mt_zzz1", "mt_zzz2"]],
    )
    fake_dl = um.create_autospec(huggingface_hub.hf_hub_download, return_value=str(parent))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_dl)
    args = argparse.Namespace(
        output_dir=str(out),
        on_policy_dir=str(out / "on_policy"),
        models=model,
        hf_prefix="issue2054_lattice/common_regen",
        parent_prefix="issue2054_lattice",
        skip_upload=True,
    )
    assert rw.stage_assist_merge(args) == 0
    merged = [
        json.loads(line)
        for line in (out / "on_policy" / model / a / fname).read_text().splitlines()
    ]
    assert {r["conv_id"] for r in merged} == set(s_ids)  # full S, no strays
    by_id = {r["conv_id"]: r["answer"] for r in merged}
    assert all(by_id[c] == "parent" for c in s_ids[:5])
    assert all(by_id[c] == "delta" for c in s_ids[5:])


# ---------------------------------------------------------------------------
# r1 Critical 1: assistant delta basis = parent REALIZED on-policy file ids
# ---------------------------------------------------------------------------
def test_export_assistant_delta_from_parent_realized_files_not_admitted(tmp_path, monkeypatch):
    """The 8,000-capped-parent shape: kept.json ADMITS all of S, but the
    parent's realized on-policy files cover only part of it — the delta must
    cover the file-missing survivors. Under the old admitted-set basis the
    delta here would be EMPTY and stage_assist_merge would abort AFTER the
    phase_c GPU spend (code-review r1 Critical 1)."""
    import unittest.mock as um

    import huggingface_hub

    out = tmp_path / "regen"
    s_ids = [f"mt_{i:04d}" for i in range(8)]
    (out / "target_set").mkdir(parents=True)
    _write_jsonl(
        out / "target_set" / "target_set.jsonl",
        [{"conv_id": c, "question": f"q {c}"} for c in s_ids],
    )
    state = {
        "artifact": "regen_wave_state",
        "admitted": {v: list(s_ids) for v in rw.CHAR_VARIANTS},
        "attempts": {v: {c: 1 for c in s_ids} for v in rw.CHAR_VARIANTS},
        "assistant_existing": list(s_ids),  # ADMITTED covers ALL of S (the false premise)
        "waves_done": [{"wave": 1}],
        "wave0": {"regime": {}},
    }
    rw._write_state(out, state)
    for v in rw.CHAR_VARIANTS:
        _write_jsonl(
            out / "scaffolds" / v / f"scaffolds_{v}.wave1_admitted.jsonl",
            [
                {"conv_id": c, "variant": v, "question": f"q {c}", "scaffold_text": "s"}
                for c in s_ids
            ],
        )
    # Parent REALIZED on-policy files: first 5 survivors only (the 8k-cap shape).
    parent = tmp_path / "parent_realized.jsonl"
    _write_jsonl(parent, [{"conv_id": c, "answer": "parent"} for c in s_ids[:5]])
    fake_dl = um.create_autospec(huggingface_hub.hf_hub_download, return_value=str(parent))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_dl)
    args = argparse.Namespace(
        output_dir=str(out),
        state_from_hf=False,
        seed=137,
        models="qwen2.5-7b-instruct,qwen2.5-7b",
        hf_prefix="issue2054_lattice/common_regen",
        parent_prefix="issue2054_lattice",
        skip_upload=True,
    )
    assert rw.stage_export(args) == 0
    a = rw.ASSISTANT_VARIANT
    delta_rows = [
        json.loads(line)
        for line in (out / "assistant_delta" / a / f"scaffolds_{a}.jsonl").read_text().splitlines()
    ]
    # Delta = S minus parent-FILE ids (3 rows) - NOT S minus admitted (empty).
    assert {r["conv_id"] for r in delta_rows} == set(s_ids[5:])
    man = json.loads((out / "scaffolds" / "export_manifest.json").read_text())
    assert man["assistant_delta"] == 3
    assert man["assistant_delta_basis"] == "parent_realized_on_policy_files"
    assert man["assistant_existing_in_s"] == 5  # realized file coverage
    assert man["assistant_admitted_in_s"] == 8  # informational: admitted overstates


def test_parent_realized_ids_use_intersection_on_cell_mismatch(tmp_path, monkeypatch):
    """Mismatched per-cell parent id-sets degrade to the INTERSECTION (delta
    is then a superset per cell, so assist-merge still reaches full S)."""
    import unittest.mock as um

    import huggingface_hub

    p_a = tmp_path / "cell_a.jsonl"
    p_b = tmp_path / "cell_b.jsonl"
    _write_jsonl(p_a, [{"conv_id": f"mt_{i:04d}"} for i in range(0, 5)])
    _write_jsonl(p_b, [{"conv_id": f"mt_{i:04d}"} for i in range(1, 6)])
    fake_dl = um.create_autospec(
        huggingface_hub.hf_hub_download,
        side_effect=[str(p_a), str(p_b), str(p_a), str(p_b)],
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_dl)
    args = argparse.Namespace(models="m1,m2", parent_prefix="issue2054_lattice")
    ids = rw._parent_assistant_realized_ids(args)
    assert ids == {f"mt_{i:04d}" for i in range(1, 5)}
    assert fake_dl.call_count == 4  # 2 models x 2 forms


def test_assist_merge_stays_fail_loud_when_delta_misses_uncovered_survivor(tmp_path, monkeypatch):
    """The pre-fix production shape (admitted-basis delta = EMPTY while the
    parent file is 8,000-capped): the merge coverage assert must still abort
    fail-loud — the export-side fix makes this branch unreachable, it does
    not weaken it (code-review r1 Critical 1, 'keep the merge assert
    unchanged')."""
    import unittest.mock as um

    import huggingface_hub

    out = tmp_path / "regen"
    s_ids = [f"mt_{i:04d}" for i in range(8)]
    (out / "scaffolds").mkdir(parents=True)
    (out / "scaffolds" / "survivor_set.json").write_text(json.dumps({"survivor_conv_ids": s_ids}))
    parent = tmp_path / "parent.jsonl"
    _write_jsonl(parent, [{"conv_id": c, "answer": "parent"} for c in s_ids[:5]])
    fake_dl = um.create_autospec(huggingface_hub.hf_hub_download, return_value=str(parent))
    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_dl)
    args = argparse.Namespace(
        output_dir=str(out),
        on_policy_dir=str(out / "on_policy"),  # no delta files written
        models="qwen2.5-7b-instruct",
        hf_prefix="issue2054_lattice/common_regen",
        parent_prefix="issue2054_lattice",
        skip_upload=True,
    )
    with pytest.raises(RuntimeError, match="under-covered at capture"):
        rw.stage_assist_merge(args)


# ---------------------------------------------------------------------------
# r1 Minor 2: gen attempts ledger idempotent per wave
# ---------------------------------------------------------------------------
def test_record_gen_wave_idempotent_on_same_wave_rerun():
    v0 = rw.CHAR_VARIANTS[0]
    state = {"admitted": {}, "attempts": {v: {} for v in rw.CHAR_VARIANTS}}
    pending = {v: [] for v in rw.CHAR_VARIANTS}
    pending[v0] = ["mt_0001", "mt_0002"]
    assert rw._record_gen_wave(state, 2, [v0], pending, {}, 139, False) is True
    assert state["attempts"][v0] == {"mt_0001": 1, "mt_0002": 1}
    # Same-wave re-run (crashed/re-run gen leg): no double increment.
    assert rw._record_gen_wave(state, 2, [v0], pending, {}, 139, False) is False
    assert state["attempts"][v0] == {"mt_0001": 1, "mt_0002": 1}
    assert len(state["gen_waves"]) == 1
    # A LATER wave still increments.
    assert rw._record_gen_wave(state, 3, [v0], pending, {}, 140, False) is True
    assert state["attempts"][v0] == {"mt_0001": 2, "mt_0002": 2}


# ---------------------------------------------------------------------------
# r1 Minor 3: judge argv must match the wave-0 recorded instrument
# ---------------------------------------------------------------------------
def test_judge_leg_refuses_argv_drift_from_wave0_regime(tmp_path):
    import issue2054_phase_a as pa

    out = tmp_path / "regen"
    (out / "target_set").mkdir(parents=True)
    _write_jsonl(out / "target_set" / "target_set.jsonl", [{"conv_id": "mt_0001", "question": "q"}])
    rubric_sha = pa.hashlib.sha256(pa._scaffold_judge_rubric().encode()).hexdigest()[:16]
    state = {
        "admitted": {v: [] for v in rw.CHAR_VARIANTS},
        "attempts": {v: {} for v in rw.CHAR_VARIANTS},
        "waves_done": [],
        "wave0": {"regime": {"rubric_sha256": rubric_sha, "threshold": 50.0, "n_draws": 1}},
    }
    rw._write_state(out, state)
    args = argparse.Namespace(
        output_dir=str(out),
        state_from_hf=False,
        wave=1,
        judge_keep_threshold=60.0,  # drifted vs the recorded 50.0
        judge_draws=1,
        max_tokens=1024,
        pilot_n=200,
        skip_pilot=False,
        prejudge_from_hf=False,
        skip_upload=True,
        seed=137,
        hf_prefix="issue2054_lattice/common_regen",
        parent_prefix="issue2054_lattice",
    )
    with pytest.raises(RuntimeError, match="drifts from the wave-0 recorded instrument"):
        rw.stage_judge(args)


def test_stage_r2_inputs_body_stages_every_consumer_input(tmp_path, monkeypatch):
    import unittest.mock as um

    from explore_persona_space.orchestrate import hub

    staged: list[str] = []

    def fake_stage_hub_file(repo_id, path_in_repo, target, repo_type=None, overwrite=False):
        staged.append(path_in_repo)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text("{}")
        return Path(target)

    fake_file = um.create_autospec(hub.stage_hub_file, side_effect=fake_stage_hub_file)
    monkeypatch.setattr(hub, "stage_hub_file", fake_file)

    sharded: list[str] = []
    monkeypatch.setattr(
        rw,
        "_stage_sharded_or_plain",
        lambda prefix, stem, dest: sharded.append(f"{prefix}/{stem}") or (dest / f"{stem}.jsonl"),
    )
    args = argparse.Namespace(
        output_dir=str(tmp_path / "regen"),
        hf_prefix="issue2054_lattice/common_regen",
        fold_map_out=str(tmp_path / "fold_map_extended.json"),
    )
    assert rw.stage_r2_inputs(args) == 0
    # Every R2 consumer input is in the staged set (#1482 cross-machine seam).
    assert any(s.endswith("scaffolds/char_helios/scaffolds_char_helios") for s in sharded)
    assert any(
        s.endswith(f"scaffolds/{rw.ASSISTANT_VARIANT}/scaffolds_{rw.ASSISTANT_VARIANT}")
        for s in sharded
    )
    assert any("assistant_delta" in s for s in sharded)
    assert any(s.endswith("answers/answers_pool") for s in sharded)
    assert any(s.endswith("scaffolds/survivor_set.json") for s in staged)
    assert any(s.endswith("scaffolds/export_manifest.json") for s in staged)
    assert any(s.endswith("state/shared_fold_map_extended.json") for s in staged)
    assert (tmp_path / "fold_map_extended.json").exists()
