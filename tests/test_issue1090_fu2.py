"""Offline tests for the #1090 ``fu2-dose-extension`` follow-up driver.

Covers (followup-scope brief): the epochs-6 config construction (verbatim
recipe values except epochs), ladder-rung enumeration 2..30, band selection
on synthetic curves (in-band pick vs closest_approach), overflow routing of
the adapter upload, and the frozen-mix sha-verification refusal on mismatch.

The upload boundary is faked with a signature-conformant recorder (mirrors
``hub._upload``'s positional/keyword surface); every other test executes the
PRODUCTION bodies (verify_staged_mix / build_smoke_mix_fixture /
enumerate_ckpt_rungs / select_fu2_dose / phase_fu2_upload / fu2_regime_key).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _fu2_module():
    """Import scripts/issue1090_fu2.py (self-inserts scripts/ on sys.path)."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts import issue1090_fu2 as fu2
    finally:
        sys.path.remove(str(REPO_ROOT))
    return fu2


def _run_module():
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts import issue1090_run as run
    finally:
        sys.path.remove(str(REPO_ROOT))
    return run


# ── 1. epochs-6 config construction: THE single recipe deviation ─────────────


def test_fu2_train_config_differs_from_parent_only_in_epochs():
    """Build the PARENT production TrainLoraConfig (recipe_for + the
    build_organism recipe_max_length seam + the _make_train_fn save_steps /
    max_length pins) and the fu2 config through the SAME seams; the field
    diff must be exactly {epochs: 3 -> 6}."""
    fu2 = _fu2_module()
    run = _run_module()
    from explore_persona_space.artifacts.recipe import build_train_config, recipe_for

    parent_spec = recipe_for("sycophancy", arm="primary")
    parent_spec = dataclasses.replace(
        parent_spec,
        overrides={**parent_spec.overrides, "max_length": run.MAX_LENGTH_1090},
    )
    parent_cfg = build_train_config(parent_spec, run_name="x", seed=42)
    parent_cfg = dataclasses.replace(
        parent_cfg, save_steps=run.SAVE_STEPS_1090, max_length=run.MAX_LENGTH_1090
    )

    fu2_cfg = build_train_config(fu2.fu2_recipe_spec(), run_name="x", seed=42)
    fu2_cfg = dataclasses.replace(
        fu2_cfg, save_steps=run.SAVE_STEPS_1090, max_length=run.MAX_LENGTH_1090
    )

    parent_d = dataclasses.asdict(parent_cfg)
    fu2_d = dataclasses.asdict(fu2_cfg)
    diff = {k for k in parent_d if parent_d[k] != fu2_d[k]}
    assert diff == {"epochs"}, f"fu2 config deviates beyond epochs: {sorted(diff)}"
    assert parent_cfg.epochs == 3
    assert fu2_cfg.epochs == fu2.FU2_EPOCHS == 6
    # The parent's own declared deviations carried verbatim.
    assert fu2_cfg.save_steps == 2
    assert fu2_cfg.max_length == 2048


def test_fu2_recipe_spec_keeps_stopping_and_mix_knobs():
    fu2 = _fu2_module()
    from explore_persona_space.artifacts.recipe import JUDGED_RATE_BAND, recipe_for

    spec = fu2.fu2_recipe_spec()
    parent = recipe_for("sycophancy", arm="primary")
    assert spec.stopping == parent.stopping
    assert spec.stopping.rate_band == JUDGED_RATE_BAND == (0.60, 0.85)
    assert (spec.generic_frac, spec.neg_ratio, spec.arm, spec.train_method) == (
        parent.generic_frac,
        parent.neg_ratio,
        parent.arm,
        parent.train_method,
    )


def test_fu2_cell_run_name_distinct_from_parent():
    fu2 = _fu2_module()
    run = _run_module()
    c3 = fu2.resolve_fu2_cells(None, True)[0]
    assert c3.slug == run.CELL_BY_ID["c3"].slug  # same artifact identity
    assert c3.run_name != run.CELL_BY_ID["c3"].run_name  # fresh WandB run
    assert c3.run_name.startswith("issue1090_fu2_")
    with pytest.raises(ValueError, match="bad fu2 cells"):
        fu2.resolve_fu2_cells("c1", False)


# ── 2. Ladder-rung enumeration 2..30 ─────────────────────────────────────────


def test_expected_rungs_80_rows_is_2_to_30():
    fu2 = _fu2_module()
    rungs, total = fu2.fu2_expected_rungs(80)
    assert total == 30  # 80 rows / (4*4) = 5 steps/epoch * 6 epochs
    assert rungs == list(range(2, 31, 2))


def test_expected_rungs_small_mix():
    fu2 = _fu2_module()
    rungs, total = fu2.fu2_expected_rungs(8)  # ceil(8/16)=1 step/epoch * 6
    assert total == 6
    assert rungs == [2, 4, 6]


def test_enumerate_ckpt_rungs_numeric_and_failloud(tmp_path):
    fu2 = _fu2_module()
    for step in (2, 4, 10, 25, 30, 100):
        (tmp_path / f"checkpoint-{step}").mkdir()
    (tmp_path / "checkpoint-final").mkdir()  # non-numeric suffix: ignored
    (tmp_path / "checkpoint-99").write_text("")  # file, not dir: ignored
    ckpts = fu2.enumerate_ckpt_rungs(tmp_path)
    assert sorted(ckpts) == [2, 4, 10, 25, 30, 100]  # numeric, never lexical
    with pytest.raises(ValueError, match="no checkpoint-<step> dirs"):
        fu2.enumerate_ckpt_rungs(tmp_path / "empty-does-not-exist")


# ── 3. Band selection on synthetic curves ────────────────────────────────────


def test_band_selection_in_band_pick_is_earliest():
    fu2 = _fu2_module()
    # Rising curve: enters [0.60, 0.85] at step 18.
    rates = {2: 0.10, 6: 0.30, 10: 0.55, 14: 0.58, 18: 0.62, 22: 0.80, 26: 0.90, 30: 0.95}
    sel = fu2.select_fu2_dose(rates)
    assert (sel.step, sel.in_band, sel.fallback) == (18, True, None)
    assert sel.rate == 0.62


def test_band_selection_closest_approach_when_never_in_band():
    fu2 = _fu2_module()
    # Still-rising-but-below-band curve (the fu1/parent shape): closest to
    # the band edge 0.60 is 0.549 -> fallback, never a fake band entry.
    rates = {2: 0.05, 10: 0.549, 14: 0.51, 30: 0.40}
    sel = fu2.select_fu2_dose(rates)
    assert (sel.step, sel.in_band, sel.fallback) == (10, False, "closest_approach")


def test_band_selection_overshoot_between_rungs_falls_back():
    fu2 = _fu2_module()
    # Jumps over the band between rungs: 0.55 -> 0.90; closest is 0.55 (0.05
    # below lo) vs 0.90 (0.05 above hi) -> distance tie resolves EARLIEST.
    rates = {2: 0.55, 4: 0.90}
    sel = fu2.select_fu2_dose(rates)
    assert (sel.step, sel.in_band, sel.fallback) == (2, False, "closest_approach")


# ── 4. Overflow routing of the adapter upload ────────────────────────────────


def _recording_upload(calls):
    """Signature-conformant fake of hub._upload's call surface."""

    def upload(local_path, repo_id, repo_type, path_in_repo, **kw):
        calls.append(
            {
                "local_path": str(local_path),
                "repo_id": repo_id,
                "repo_type": repo_type,
                "path_in_repo": path_in_repo,
                **{k: str(v) for k, v in kw.items()},
            }
        )
        return f"test://{repo_id}/{path_in_repo}"

    return upload


def _seed_fu2_tree(fu2, run, tmp_path, cell):
    cell_root = tmp_path / cell.slug
    (cell_root / "train" / "checkpoint-2").mkdir(parents=True)
    (cell_root / "train" / "checkpoint-2" / "adapter_config.json").write_text("{}")
    (cell_root / "rate" / "rate_checkpoint-2").mkdir(parents=True)
    (cell_root / "rate" / "rate_checkpoint-2" / "x.json").write_text("{}")
    (cell_root / "mix").mkdir(parents=True)
    (cell_root / "mix" / "mix_verification.json").write_text("{}")
    (cell_root / "fu2_build_result.json").write_text("{}")
    (cell_root / "fu2_ladder.json").write_text("{}")
    (tmp_path / "tier2" / cell.slug).mkdir(parents=True)
    (tmp_path / "tier2" / cell.slug / "y.json").write_text("{}")
    (tmp_path / "fu2_run_config.json").write_text("{}")
    return cell_root


def test_phase_fu2_upload_routes_adapters_to_overflow_only(tmp_path):
    fu2 = _fu2_module()
    run = _run_module()
    from explore_persona_space.orchestrate import hub

    cell = fu2.resolve_fu2_cells("c3", False)[0]
    _seed_fu2_tree(fu2, run, tmp_path, cell)
    calls: list[dict] = []
    cfg = run.RunConfig(smoke=False, cells=(cell,), out_root=tmp_path)
    seams = run.Seams1090(upload_fn=_recording_upload(calls))
    records = {cell.slug: {"status": "trained"}}

    uploaded = fu2.phase_fu2_upload(cfg, seams, records)

    model_calls = [c for c in calls if c["repo_type"] == "model"]
    assert len(model_calls) == 1
    assert model_calls[0]["repo_id"] == hub.DEFAULT_OVERFLOW_REPO
    assert model_calls[0]["path_in_repo"] == f"issue1090/fu2/{cell.slug}"
    assert model_calls[0]["private"] == "True"  # never created public (#564)
    # NEVER the canonical model repo (100k-file limit — followup-scope directive).
    assert all(c["repo_id"] != run.HF_MODEL_REPO for c in calls)
    # Every text/JSON artifact rides the data repo under the fu2 prefix.
    dataset_calls = [c for c in calls if c["repo_type"] == "dataset"]
    assert dataset_calls, "no dataset uploads recorded"
    for c in dataset_calls:
        assert c["repo_id"] == run.HF_DATA_REPO
        assert c["path_in_repo"].startswith(f"{run.DATA_PREFIX}/{fu2.FU2_LABEL}")
    assert f"issue1090/fu2/{cell.slug}" in uploaded


def test_phase_fu2_upload_skips_adapter_for_untrained_cell(tmp_path):
    fu2 = _fu2_module()
    run = _run_module()
    cell = fu2.resolve_fu2_cells("c3", False)[0]
    _seed_fu2_tree(fu2, run, tmp_path, cell)
    calls: list[dict] = []
    cfg = run.RunConfig(smoke=False, cells=(cell,), out_root=tmp_path)
    seams = run.Seams1090(upload_fn=_recording_upload(calls))
    fu2.phase_fu2_upload(cfg, seams, {cell.slug: {"status": "skipped"}})
    assert not [c for c in calls if c["repo_type"] == "model"]


def test_phase_fu2_upload_fails_loud_on_empty_return(tmp_path):
    fu2 = _fu2_module()
    run = _run_module()
    cell = fu2.resolve_fu2_cells("c3", False)[0]
    _seed_fu2_tree(fu2, run, tmp_path, cell)
    cfg = run.RunConfig(smoke=False, cells=(cell,), out_root=tmp_path)
    seams = run.Seams1090(upload_fn=lambda *a, **kw: "")  # the hub fail-soft ""
    with pytest.raises(RuntimeError, match="upload returned no path"):
        fu2.phase_fu2_upload(cfg, seams, {cell.slug: {"status": "trained"}})


# ── 5. Frozen-mix sha verification ───────────────────────────────────────────


def _write_mix(tmp_path, rows, meta):
    d = tmp_path / "mix"
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "train_mix.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    (d / "mix_meta.json").write_text(json.dumps(meta))
    return d


def test_verify_staged_mix_refuses_sha_mismatch(tmp_path):
    fu2 = _fu2_module()
    cell = fu2.resolve_fu2_cells("c3", False)[0]
    rows = [{"prompt": [], "completion": []}] * 4
    meta = {
        "counts_realized": {"positives": 2, "negatives": 2, "generic": 0},
        "spec": {"behavior_name": "sycophancy"},
        "train_mix_sha256": "0" * 64,  # wrong pin
    }
    d = _write_mix(tmp_path, rows, meta)
    with pytest.raises(ValueError, match="sha256 mismatch"):
        fu2.verify_staged_mix(d, cell)


def test_verify_staged_mix_accepts_correct_pin_and_records(tmp_path):
    fu2 = _fu2_module()
    from explore_persona_space.artifacts.organisms import _sha256_file

    cell = fu2.resolve_fu2_cells("c3", False)[0]
    rows = [{"prompt": [], "completion": []}] * 4
    meta = {
        "counts_realized": {"positives": 2, "negatives": 2, "generic": 0},
        "spec": {"behavior_name": "sycophancy"},
    }
    d = _write_mix(tmp_path, rows, meta)
    sha = _sha256_file(d / "train_mix.jsonl")
    meta["train_mix_sha256"] = sha
    (d / "mix_meta.json").write_text(json.dumps(meta))
    rec = fu2.verify_staged_mix(d, cell)
    assert rec["train_mix_sha256"] == sha
    assert rec["sha_pinned_in_meta"] is True
    assert rec["n_rows"] == 4
    assert (d / "mix_verification.json").exists()


def test_verify_staged_mix_refuses_row_count_mismatch(tmp_path):
    fu2 = _fu2_module()
    cell = fu2.resolve_fu2_cells("c3", False)[0]
    rows = [{"prompt": [], "completion": []}] * 3  # meta says 4
    meta = {
        "counts_realized": {"positives": 2, "negatives": 2, "generic": 0},
        "spec": {"behavior_name": "sycophancy"},
    }
    d = _write_mix(tmp_path, rows, meta)
    with pytest.raises(ValueError, match="row-count mismatch"):
        fu2.verify_staged_mix(d, cell)


def test_verify_staged_mix_refuses_wrong_behavior(tmp_path):
    fu2 = _fu2_module()
    cell = fu2.resolve_fu2_cells("c3", False)[0]
    rows = [{"prompt": [], "completion": []}] * 2
    meta = {
        "counts_realized": {"positives": 2},
        "spec": {"behavior_name": "impolite"},
    }
    d = _write_mix(tmp_path, rows, meta)
    with pytest.raises(ValueError, match="behavior"):
        fu2.verify_staged_mix(d, cell)


def test_smoke_fixture_passes_the_same_verify_gate(tmp_path):
    """The smoke fixture is written in the production on-disk shape and PINS
    its sha — the SAME verify branch runs in smoke and full (PASS_UNIFIED)."""
    fu2 = _fu2_module()
    run = _run_module()
    cell = fu2.resolve_fu2_cells(None, True)[0]
    cfg = run.RunConfig(smoke=True, cells=(cell,), out_root=tmp_path)
    fu2.build_smoke_mix_fixture(cfg, cell)
    rec = fu2.verify_staged_mix(fu2._cell_mix_dir(cfg, cell), cell)
    assert rec["sha_pinned_in_meta"] is True
    assert rec["n_rows"] == 8


# ── Regime key: the fu2 deviations enter the resume key ──────────────────────


def test_fu2_regime_key_carries_epochs_and_judge_budget(tmp_path):
    fu2 = _fu2_module()
    run = _run_module()
    cell = fu2.resolve_fu2_cells(None, True)[0]
    cfg = run.RunConfig(smoke=True, cells=(cell,), out_root=tmp_path)
    key = fu2.fu2_regime_key(cfg)
    assert key["fu2_epochs"] == 6
    assert key["fu2_judge_max_tokens"] == 300
    assert key["followup_label"] == "fu2-dose-extension"
    assert key["max_length"] == 2048
