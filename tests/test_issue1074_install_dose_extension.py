"""Offline tests for the #1074 follow-up `install-dose-extension` surfaces.

Covers the held-out margin-pool derivation (the plan v9 §2 measurement fix —
trained-row exclusion by matching against the pinned mix, the fail-loud
untrained-count gate, seeded sampling), the label-keyed CLI/config/regime-key
behavior, the `-e9` upload routing (never clobber the parent 3-epoch ladder),
and the aggregate additions (drop-censoring telemetry + the schedule-stretch
overlay + the dose-extension run_followup branch).

All model/judge boundaries are the sanctioned injected stubs (no network); the
end-to-end body coverage of the driver pipeline is the `--smoke --followup
install-dose-extension` run recorded in the implementation report.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from explore_persona_space.eval.graded_judge import JudgeResult

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import issue1074_aggregate as aggregate  # noqa: E402
import issue1074_generator_compare as driver  # noqa: E402

LABEL = driver.LABEL_DOSE_EXTENSION
CELL = driver.MIXED_CELL_SLUG


# ── fixtures: staged pinned files shaped like the REAL artifacts ─────────────


def _write_jsonl(path: Path, rows) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return path


def _candidate(arm: str, i: int, *, completion: str | None) -> dict:
    return {
        "request_id": f"{arm}-{i:05d}",
        "arm": arm,
        "question_id": f"q{i:03d}",
        "variant_id": f"v{i % 3}",
        "question": f"question {i}?",
        "completion": completion,
        "drop_reason": None if completion is not None else "empty",
    }


def _staged_fixture(
    tmp_path: Path,
    *,
    n_pos_kept: int = 6,
    n_pos_trained: int = 4,
    n_neg_kept: int = 5,
    n_neg_trained: int = 3,
) -> dict[str, Path]:
    """Synthetic staged set mirroring the real shapes: raw_pos/raw_neg rows +
    judge_rows kept flags + a train_mix whose assistant contents are the
    TRAINED candidates' completions verbatim (datagen._train_row)."""
    pos = [_candidate("positive", i, completion=f"pos text {i}") for i in range(n_pos_kept)]
    pos.append(_candidate("positive", 90, completion=None))  # gen-dropped, never judged
    pos.append(_candidate("positive", 91, completion="pos not kept"))
    neg = [_candidate("negative", i, completion=f"neg text {i}") for i in range(n_neg_kept)]
    neg.append(_candidate("negative", 92, completion="neg not kept"))
    judge_rows = [
        {
            "request_id": r["request_id"],
            "question_id": r["question_id"],
            "variant_id": r["variant_id"],
            "arm": r["arm"],
            "scores": [90.0],
            "mean": 90.0,
            "kept": r["request_id"] not in ("positive-00091", "negative-00092"),
        }
        for r in [*pos, *neg]
        if r["completion"] is not None
    ]
    mix_rows = [
        {
            "prompt": [{"role": "user", "content": "q"}],
            "completion": [{"role": "assistant", "content": f"pos text {i}"}],
        }
        for i in range(n_pos_trained)
    ] + [
        {
            "prompt": [{"role": "user", "content": "q"}],
            "completion": [{"role": "assistant", "content": f"neg text {i}"}],
        }
        for i in range(n_neg_trained)
    ]
    mix_rows.append(  # a generic row (never matches a candidate completion)
        {
            "prompt": [{"role": "user", "content": "generic"}],
            "completion": [{"role": "assistant", "content": "a generic answer"}],
        }
    )
    return {
        "mix/train_mix.jsonl": _write_jsonl(tmp_path / "mix" / "train_mix.jsonl", mix_rows),
        "mix/mix_meta.json": _write_jsonl(tmp_path / "mix" / "mix_meta.json", [{"ok": True}]),
        "datagen/raw_pos.jsonl": _write_jsonl(tmp_path / "datagen" / "raw_pos.jsonl", pos),
        "datagen/raw_neg.jsonl": _write_jsonl(tmp_path / "datagen" / "raw_neg.jsonl", neg),
        "datagen/judge_rows.jsonl": _write_jsonl(
            tmp_path / "datagen" / "judge_rows.jsonl", judge_rows
        ),
    }


# ── held-out margin-pool derivation ──────────────────────────────────────────


def test_heldout_pools_trained_exclusion_and_sampling(tmp_path):
    staged = _staged_fixture(tmp_path)
    pos_pairs, neg_pairs, prov = driver.derive_heldout_margin_pools(
        staged, expected_untrained={"positive": 2, "negative": 2}, pool_n=25, seed=42
    )
    # kept-but-untrained only: trained texts + not-kept + gen-dropped all excluded.
    assert {p["answer"] for p in pos_pairs} == {"pos text 4", "pos text 5"}
    assert {p["answer"] for p in neg_pairs} == {"neg text 3", "neg text 4"}
    assert prov["counts"]["positive"] == {
        "kept": 6,
        "trained_matched": 4,
        "untrained": 2,
        "sampled": 2,
    }
    assert prov["pool_seed"] == 42 and prov["kind"] == "heldout_untrained"
    # Pair shape matches organisms.derive_margin_pools (margin_fn contract).
    assert set(pos_pairs[0]) == {"probe", "answer", "question_id", "variant_id", "request_id"}
    # Deterministic across calls (seeded sample + stable sort).
    again = driver.derive_heldout_margin_pools(
        staged, expected_untrained={"positive": 2, "negative": 2}, pool_n=25, seed=42
    )
    assert again[0] == pos_pairs and again[1] == neg_pairs


def test_heldout_pools_pool_n_cap(tmp_path):
    staged = _staged_fixture(tmp_path)
    pos_pairs, neg_pairs, prov = driver.derive_heldout_margin_pools(
        staged, expected_untrained={"positive": 2, "negative": 2}, pool_n=1, seed=42
    )
    assert len(pos_pairs) == 1 and len(neg_pairs) == 1
    assert prov["counts"]["positive"]["sampled"] == 1
    assert prov["pool_n_requested"] == 1


def test_heldout_pools_fail_loud_on_count_drift(tmp_path):
    staged = _staged_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="untrained count 2 != expected 64"):
        driver.derive_heldout_margin_pools(staged)  # real 64/30 gate vs tiny fixture
    with pytest.raises(RuntimeError, match="negative untrained count 2 != expected 3"):
        driver.derive_heldout_margin_pools(
            staged, expected_untrained={"positive": 2, "negative": 3}
        )


# ── CLI / config / regime key ────────────────────────────────────────────────


def test_dose_extension_cli_and_config(tmp_path):
    args = driver._parse_args(["--followup", LABEL])
    assert args.full and not args.smoke  # --followup implies --full
    cfg = driver.config_from_args(args)
    assert cfg.followup_label == LABEL
    assert [c.slug for c in cfg.cells] == [CELL]
    # Label-keyed smoke scratch root: no cross-label regime collision in /tmp.
    smoke_cfg = driver.config_from_args(driver._parse_args(["--smoke", "--followup", LABEL]))
    bnr_cfg = driver.config_from_args(
        driver._parse_args(["--smoke", "--followup", driver.LABEL_BASE_NEG_REGEN])
    )
    assert smoke_cfg.out_root != bnr_cfg.out_root
    # --resume-partial-attempt is base-negatives-regen-only (no datagen here).
    with pytest.raises(SystemExit):
        driver._parse_args(["--followup", LABEL, "--resume-partial-attempt", "att-x"])


def test_dose_extension_regime_key(tmp_path):
    cfg = driver.config_from_args(driver._parse_args(["--followup", LABEL]))
    cfg.out_root = tmp_path
    with pytest.raises(RuntimeError, match="requires the staged pinned mix"):
        cfg.regime_key()  # fail-loud before staging
    mix = tmp_path / "train_mix.jsonl"
    mix.write_text('{"row": 1}\n')
    cfg.pinned_mix = mix
    key = cfg.regime_key()
    assert key["followup_label"] == LABEL
    assert key["epochs_override"] == driver.DOSE_EXT_EPOCHS == 9
    assert key["pinned_mix"]["revision"] == driver.MIX_PIN_REVISION
    assert len(key["pinned_mix"]["sha256"]) == 64
    # The BNR round's key is UNCHANGED (no dose keys).
    bnr = driver.config_from_args(driver._parse_args(["--followup", driver.LABEL_BASE_NEG_REGEN]))
    bnr_key = bnr.regime_key()
    assert "epochs_override" not in bnr_key and "pinned_mix" not in bnr_key


def test_run_name_and_model_prefix_label_keying():
    cfg = driver.config_from_args(driver._parse_args(["--followup", LABEL]))
    (cell,) = cfg.cells
    assert driver._run_name_for(cfg, cell) == "issue1074_harmful_compliance_mixed_e9_seed42"
    assert driver._cell_model_prefix(cfg, cell) == "issue1074/harmful_compliance-mixed-e9"
    bnr = driver.config_from_args(driver._parse_args(["--followup", driver.LABEL_BASE_NEG_REGEN]))
    assert driver._run_name_for(bnr, cell) == cell.run_name
    assert driver._cell_model_prefix(bnr, cell) == "issue1074/harmful_compliance-mixed"


# ── upload routing (never clobber the parent round's paths) ──────────────────


def test_phase_upload_dose_extension_routing(tmp_path):
    cfg = driver.config_from_args(driver._parse_args(["--smoke", "--followup", LABEL]))
    cfg.out_root = tmp_path
    cell_root = tmp_path / CELL
    (cell_root / "train").mkdir(parents=True)
    (cell_root / "train" / "adapter.bin").write_text("x")
    (cell_root / "rate").mkdir()
    (cell_root / "rate" / "r.json").write_text("{}")
    (cell_root / "build_result.json").write_text('{"status": "trained"}')
    (tmp_path / "evalgen").mkdir()
    (tmp_path / "evalgen" / "m.json").write_text("{}")
    (tmp_path / "heldout_margin_pools.json").write_text("{}")
    (tmp_path / "run_config.json").write_text("{}")
    calls: list[tuple] = []

    def recording_upload(local_path, repo_id, repo_type, path_in_repo, **kw) -> str:
        calls.append((repo_id, path_in_repo))
        return f"smoke://{repo_id}/{path_in_repo}"

    seams = driver.Seams1074(upload_fn=recording_upload)
    uploaded = driver.phase_upload(cfg, seams, {CELL: {"status": "trained"}})
    paths = {p for _, p in calls}
    run_prefix = f"{driver.DATA_PREFIX}/followups/{LABEL}"
    # Adapter ladder -> the -e9 model prefix (parent 3-epoch ladder untouched).
    assert ("superkaiba1/explore-persona-space", f"issue1074/{CELL}-e9") in calls
    # Rate completions -> the -e9 rate prefix.
    assert f"{driver.DATA_PREFIX}/raw_completions/rate/{CELL}-e9" in paths
    # Cell-level files -> the followup run prefix, NEVER the parent cell path.
    assert f"{run_prefix}/{CELL}/build_result.json" in paths
    assert not any(p.startswith(f"{driver.DATA_PREFIX}/{CELL}/") for p in paths)
    # Run-level: final completions + held-out pools under the followup prefix.
    assert f"{run_prefix}/raw_completions/final" in paths
    assert f"{run_prefix}/heldout_margin_pools.json" in paths
    assert uploaded  # manifest recorded


# ── aggregate: overlay + drop-censoring + the dose run_followup branch ───────


def test_dose_overlay(tmp_path):
    prior = tmp_path / "prior.json"
    prior.write_text(
        json.dumps({"dose_curve_rates_by_step": {"25": 0.124, "50": 0.154, "90": 0.260}})
    )
    out = aggregate.dose_overlay({"25": 0.20, "50": 0.30, "270": 0.70}, prior)
    assert out["status"] == "computed"
    assert out["shared_steps"] == ["25", "50"]
    assert out["delta_at_shared_steps"]["25"] == pytest.approx(0.076)
    with pytest.raises(RuntimeError, match="prior install summary missing"):
        aggregate.dose_overlay({"25": 0.2}, tmp_path / "nope.json")


def _write_rate_tree(base: Path, step: int, *, n_q: int = 2, drop_item: bool = False) -> None:
    d = base / f"rate_checkpoint-{step}"
    (d / "judge" / "trained_persona_software_engineer").mkdir(parents=True)
    payload = {
        "questions": [f"q{i}" for i in range(n_q)],
        "completions": [[f"c{i}"] for i in range(n_q)],
        "manifest": {"questions_sha256": "x"},
    }
    (d / "completions__trained__persona_software_engineer.json").write_text(json.dumps(payload))
    all_scores = {}
    for i in range(n_q):
        for draw in range(2):
            cid = f"ctx-trained-q{i:03d}-c0__{i:05d}__{draw:02d}"
            dropped = drop_item and i == 0
            all_scores[cid] = {"score": None} if dropped else {"score": 80}
    (d / "judge" / "trained_persona_software_engineer" / "judge_raw.json").write_text(
        json.dumps({"all_scores": all_scores})
    )


def test_dose_rate_drop_censoring_both_layouts(tmp_path):
    # Local driver out_root layout.
    local_root = tmp_path / "local"
    _write_rate_tree(local_root / CELL / "rate", 25)
    _write_rate_tree(local_root / CELL / "rate", 50, drop_item=True)
    per_step = aggregate.dose_rate_drop_censoring(local_root)
    assert set(per_step) == {"25", "50"}
    assert per_step["25"]["n_questions"] == 2
    assert per_step["25"]["n_scored"] == 2 and per_step["25"]["n_dropped"] == 0
    assert per_step["50"]["n_dropped"] == 1 and per_step["50"]["n_dropped_draws"] == 2
    # HF-staged layout (the -e9 rate prefix) wins when present.
    staged_root = tmp_path / "staged"
    _write_rate_tree(staged_root / "raw_completions" / "rate" / f"{CELL}-e9", 75)
    per_step2 = aggregate.dose_rate_drop_censoring(staged_root)
    assert set(per_step2) == {"75"}
    # Empty tree -> empty dict (warned, never a crash).
    assert aggregate.dose_rate_drop_censoring(tmp_path / "empty") == {}


def _stub_judge_graded(
    items,
    eval_prompt,
    *,
    n_draws,
    cache_dir,
    save_raw,
    judge_model="stub",
    temperature=1.0,
    dry_run=False,
):
    """Signature-conformant judge stub (mirrors eval.graded_judge.judge_graded)."""
    scores = {item_id: 80.0 for item_id, _q, _c in items}
    return JudgeResult(scores=scores, n_total_draws=len(items) * n_draws, n_dropped_draws=0)


def test_run_followup_dose_extension_local_root(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate, "judge_graded", _stub_judge_graded)
    root = tmp_path / "root"
    # build_result with the dose ladder provenance.
    build = {
        "status": "trained",
        "adapter_path": "train/checkpoint-270",
        "selection": {"step": 270, "rate": 0.7, "in_band": True, "fallback": None},
        "provenance": {"rates_by_step": {"25": 0.2, "50": 0.3, "270": 0.7}},
    }
    (root / CELL).mkdir(parents=True)
    (root / CELL / "build_result.json").write_text(json.dumps(build))
    # Final-eval completions (local evalgen layout).
    beh_dir = root / "evalgen" / aggregate.FOLLOWUP_BEHAVIOR
    beh_dir.mkdir(parents=True)
    payload = {
        "questions": ["q0", "q1"],
        "completions": [["a", "b"], ["c", "d"]],
        "manifest": {"questions_sha256": "sha"},
    }
    for state in (CELL, "base"):
        for ctx in ("persona_software_engineer", "neg_default_assistant"):
            (beh_dir / f"completions__{state}__{ctx}.json").write_text(json.dumps(payload))
    # Rate ladder tree + margin file + prior curve.
    _write_rate_tree(root / CELL / "rate", 25)
    (root / "margin").mkdir()
    (root / "margin" / f"{aggregate.FOLLOWUP_BEHAVIOR}.json").write_text(
        json.dumps(
            {
                "status": "computed",
                "pool_source_cell": "heldout_untrained@rev",
                "pool_provenance": {"kind": "heldout_untrained"},
                "n_pos": 25,
                "n_neg": 25,
                "cells": {f"{CELL}__persona_software_engineer": {"margin": 0.1}},
            }
        )
    )
    prior = tmp_path / "prior.json"
    prior.write_text(json.dumps({"dose_curve_rates_by_step": {"25": 0.124, "50": 0.154}}))
    args = SimpleNamespace(
        followup=LABEL,
        results_root=str(root),
        stage_dir=str(tmp_path / "stage"),
        out_dir=str(tmp_path / "out"),
        n_judge_draws=2,
        n_bootstrap=10,
        prior_install_summary=str(prior),
    )
    assert aggregate.run_followup(args) == 0
    out = tmp_path / "out" / LABEL
    summary = json.loads((out / "install" / "install_summary.json").read_text())
    assert summary["dose_curve_rates_by_step"] == {"25": 0.2, "50": 0.3, "270": 0.7}
    assert summary["overlay"]["shared_steps"] == ["25", "50"]
    assert summary["drop_censoring"]["per_checkpoint"]["25"]["n_scored"] == 2
    cells = summary["drop_censoring"]["final_cells"]
    assert cells[f"{CELL}__persona_software_engineer"]["n_questions"] == 2
    assert cells[f"{CELL}__persona_software_engineer"]["n_dropped"] == 0
    margin = json.loads((out / "margin" / "margin_summary.json").read_text())
    assert margin["pool_provenance"] == {"kind": "heldout_untrained"}
    # No BNR-only outputs on this label.
    assert not (out / "negative_yield.json").exists()


def test_run_followup_dose_extension_fail_loud_no_build(tmp_path):
    args = SimpleNamespace(
        followup=LABEL,
        results_root=str(tmp_path / "emptyroot"),
        stage_dir=str(tmp_path / "stage"),
        out_dir=str(tmp_path / "out"),
        n_judge_draws=2,
        n_bootstrap=10,
        prior_install_summary=str(tmp_path / "prior.json"),
    )
    (tmp_path / "emptyroot").mkdir()
    with pytest.raises(RuntimeError, match=r"no build_result\.json"):
        aggregate.run_followup(args)
