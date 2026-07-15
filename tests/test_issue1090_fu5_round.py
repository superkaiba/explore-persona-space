"""#1090 fu5 round (`finish-impolite-bare-and-formatting-rank`, plan v7) pins.

Permanent invariants of the fu5 extension of the shared fu4 driver
(scripts/issue1090_fu4.py) + the source-parametrized vLLM engine slot width
(organisms.py `max_lora_rank`):

- the round registry (FU5_RUNS matrix, per-round names/prefixes/paths,
  fu4 byte-compat under the default round);
- rank threading end-to-end: Fu4Run.lora_r/lora_alpha -> fu4_recipe_spec
  overrides -> TrainLoraConfig (r=256/alpha=64 for fmt-pers-r256) + the
  train_lora rsLoRA hardcode the alpha-with-rank policy leans on;
- the `_assert_adapter_rank` K5 gate (fails pre-fix on a wrong-rank adapter);
- the reused-r32 field-for-field aggregate copy from the COMMITTED
  fu4_ladders.json + the lattice folding all three ranks into one cell;
- the WIDENED conditional formatting judged re-read trigger (Tier-2/rung
  structural rate >= 0.30; was install delta >= +0.30) and the fu4 legacy
  trigger unchanged;
- the aggregate retrain-parity flag (ok / parity-degraded on (0.35, 0.5] /
  parity-failed > 0.5) — report-and-flag, abort unchanged at > 0.5;
- the K5 band derivation from the committed fu4 record (halt floor at the
  base/recorded midpoint);
- the A4 recipe-identity assert against the COMMITTED cell_manifest_fu4.json;
- the eval-split diagnostic majority logic (real JudgeResult at the judge
  boundary; signature-conformant fake) + per-split structural rates through
  the REAL formatting predicate.

Committed fixtures read (sparse cone `eval_results/issue_1090` in
tests/sparse_cones.txt): fu4-extended-dose-lr/{fu4_ladders.json,
cell_manifest_fu4.json}.
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue1090_fu4 as fu4  # noqa: E402

from explore_persona_space.artifacts.recipe import build_train_config  # noqa: E402
from explore_persona_space.eval.graded_judge import JudgeResult  # noqa: E402

FU4_DELIVERABLES = REPO_ROOT / "eval_results" / "issue_1090" / "fu4-extended-dose-lr"


@pytest.fixture
def fu5_round():
    """Select the fu5 registry for the test body; ALWAYS restore fu4 (the
    module default other test files rely on)."""
    fu4.set_round("fu5")
    try:
        yield fu4.ROUND
    finally:
        fu4.set_round("fu4")


# ── Round registry ───────────────────────────────────────────────────────────


def test_fu5_run_matrix_plan_d1(fu5_round):
    ids = [r.run_id for r in fu4.FU5_RUNS]
    assert ids == [
        "imp-bare-lr1e5",
        "imp-bare-lr3e5",
        "imp-bare-lr1e4",
        "fmt-pers-r128",
        "fmt-pers-r256",
    ]
    by_id = {r.run_id: r for r in fu4.FU5_RUNS}
    for rid, lr in (("imp-bare-lr1e5", 1e-5), ("imp-bare-lr3e5", 3e-5), ("imp-bare-lr1e4", 1e-4)):
        r = by_id[rid]
        assert (r.lr, r.lora_r, r.lora_alpha) == (lr, 32, 64)
        assert r.behavior == "impolite" and r.context_id == "default"
        assert r.mix_hub_prefix == "issue1090_fu3/C2-bare-con-impolite-claude"
        assert r.mix_layout == "fu3-flat"
        assert r.fu3_base_eval == "C2-bare-con-impolite-claude.json"
    for rid, rank in (("fmt-pers-r128", 128), ("fmt-pers-r256", 256)):
        r = by_id[rid]
        assert (r.lr, r.lora_r, r.lora_alpha) == (1e-4, rank, 64)  # alpha FIXED (rsLoRA)
        assert r.behavior == "formatting"
        assert r.mix_hub_prefix == "issue1090_pvdatagen/c1-formatting-claude/mix"
    assert by_id["fmt-pers-r256"].run_name == "issue1090_fu5_fmt-pers-r256_seed42"
    # round-threaded names/prefixes (plan D2 item 1)
    R = fu5_round
    assert R.manifest_name == "cell_manifest_fu5.json"
    assert R.ladders_name == "fu5_ladders.json"
    assert R.data_prefix == "issue1090_pvdatagen/fu5-finish-impolite-bare-and-formatting-rank"
    assert R.adapter_prefix == "adapters/issue1090_fu5"
    assert str(R.deliverables_dir).endswith(
        "eval_results/issue_1090/finish-impolite-bare-and-formatting-rank"
    )
    assert R.k3_parity_run_id == "imp-bare-lr1e5"
    assert R.max_lora_rank == 256
    # smoke = the K5 rank-threading run through the SAME resolver
    assert fu4.resolve_fu4_runs(None, smoke=True)[0].run_id == "fmt-pers-r256"


def test_fu4_round_default_unchanged():
    """The default registry stays fu4 — every existing invocation (and the
    rejudge back-compat consumers) is byte-identical."""
    assert fu4.ROUND.name == "fu4"
    assert len(fu4.FU4_RUNS) == 9
    assert fu4.resolve_fu4_runs(None, smoke=True)[0].run_id == "fmt-pers-lr1e5"
    spec = fu4.fu4_recipe_spec("impolite", 3e-5)
    assert spec.overrides["lora_r"] == 32 and spec.overrides["lora_alpha"] == 64
    assert fu4.ROUND.k3_parity_run_id == "fmt-pers-lr1e5"
    assert fu4.ROUND.reread_rate_floor is None and fu4.ROUND.k3_parity_degraded_floor is None
    with pytest.raises(ValueError, match="unknown round"):
        fu4.set_round("fu6")


# ── Rank threading (plan D2 item 1 + K5) ─────────────────────────────────────


def test_fmt_pers_r256_train_config_carries_rank(fu5_round):
    """The brief's unit check: the built TrainLoraConfig for fmt-pers-r256
    carries r=256/alpha=64; use_rslora is hardcoded True inside train_lora
    (recipe.py docstring + the source assert below), so r/alpha + the hardcode
    together pin the gamma=alpha/sqrt(r) regime the plan's alpha policy needs."""
    run = {r.run_id: r for r in fu4.FU5_RUNS}["fmt-pers-r256"]
    spec = fu4.fu4_recipe_spec(run.behavior, run.lr, lora_r=run.lora_r, lora_alpha=run.lora_alpha)
    cfg = build_train_config(spec, run_name=run.run_name, seed=42)
    assert (cfg.lora_r, cfg.lora_alpha, cfg.lr) == (256, 64, 1e-4)
    assert cfg.epochs == fu4.FU4_EPOCHS and cfg.save_steps == fu4.FU4_SAVE_STEPS
    import inspect

    from explore_persona_space.train.sft import train_lora

    assert "use_rslora=True" in inspect.getsource(train_lora)


def test_assert_adapter_rank_gate(fu5_round, tmp_path):
    """K5 rank-threading gate: passes on the run's r/alpha + rsLoRA; FAILS
    (pre-fix regression pin) on a rank-64 adapter_config under a rank-256 run
    — the silent-wrong-rank adapter the gate exists to catch."""
    run = {r.run_id: r for r in fu4.FU5_RUNS}["fmt-pers-r256"]
    ckpt = tmp_path / "checkpoint-5"
    ckpt.mkdir()
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"r": 256, "lora_alpha": 64, "use_rslora": True})
    )
    got = fu4._assert_adapter_rank(run, ckpt)
    assert got == {"r": 256, "lora_alpha": 64, "use_rslora": True}
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"r": 64, "lora_alpha": 64, "use_rslora": True})
    )
    with pytest.raises(ValueError, match="rank"):
        fu4._assert_adapter_rank(run, ckpt)
    (ckpt / "adapter_config.json").write_text(
        json.dumps({"r": 256, "lora_alpha": 64, "use_rslora": False})
    )
    with pytest.raises(ValueError, match="use_rslora"):
        fu4._assert_adapter_rank(run, ckpt)
    with pytest.raises(FileNotFoundError):
        fu4._assert_adapter_rank(run, tmp_path / "missing")


# ── Reused r32 copy + lattice + K5 band (committed fu4 fixtures) ─────────────


def _committed_fu4_ladders() -> dict:
    path = FU4_DELIVERABLES / "fu4_ladders.json"
    if not path.exists():
        pytest.skip("committed fu4_ladders.json absent (sparse worktree without the cone)")
    return json.loads(path.read_text())


def test_copy_reused_r32_field_for_field(fu5_round):
    src = _committed_fu4_ladders()["runs"]["fmt-pers-lr1e4"]
    out = {"runs": {}, "cells": {}}
    fu4._copy_reused_runs(out)
    entry = out["runs"]["reused_fu4_r32"]
    assert entry["run_id"] == "reused_fu4_r32"
    assert (entry["lora_r"], entry["lora_alpha"]) == (32, 64)
    # field-for-field: the plan-named fields copied verbatim from the source
    for field in ("rates_by_step", "selection", "tier2_trained", "margin", "base_tier2"):
        assert entry[field] == src[field], field
    assert entry["cell_key"] == "fmt-pers"
    assert entry["reused_from"]["run_id"] == "fmt-pers-lr1e4"
    # lattice folds the reused entry into the fmt-pers cell (all 3 ranks, 1 file)
    fu4._verdict_lattice_inputs(out)
    assert "fmt-pers" in out["cells"]
    assert (
        out["cells"]["fmt-pers"]["tier2_confirm"]["reused_fu4_r32"] == src["tier2_trained"]["rate"]
    )


def test_copy_reused_r32_fail_loud_on_missing_source(fu5_round, monkeypatch):
    bad = fu4.ReusedRun(
        entry_id="reused_fu4_r32",
        source_run_id="fmt-pers-lr1e4",
        ladders_path="/nonexistent/fu4_ladders.json",
        tier2_hf_prefix="x",
        context_id="persona_software_engineer",
        lora_r=32,
        lora_alpha=64,
        adapter_subfolder="x",
        adapter_revision="x",
    )
    monkeypatch.setattr(fu4, "ROUND", fu4.dataclasses.replace(fu4.ROUND, reused_runs=(bad,)))
    with pytest.raises(FileNotFoundError, match="reused-run source"):
        fu4._copy_reused_runs({"runs": {}, "cells": {}})


def test_k5_band_from_committed_record(fu5_round):
    ladders = _committed_fu4_ladders()
    halt_floor, warn_ceiling, src = fu4._k5_band(ladders, fu4.ROUND.reused_runs[0])
    assert src["step"] == 20
    assert src["recorded_rate"] == pytest.approx(0.29333, abs=1e-4)
    assert src["base_rate"] == pytest.approx(1 / 6, abs=1e-3)
    # HALT floor = midpoint of the committed trained/base band (discriminating
    # placement between the correct read (~0.293) and adapter-not-applied
    # (~0.167) bands — artifact-reuse § gate calibration)
    assert halt_floor == pytest.approx((src["base_rate"] + src["recorded_rate"]) / 2, abs=1e-9)
    assert warn_ceiling == pytest.approx(src["recorded_rate"] + 0.10, abs=1e-9)


# ── Widened reread trigger + retrain-parity flag ─────────────────────────────


def _fmt_entry(rates: dict[str, float], base_rate: float, t2_rate: float | None) -> dict:
    e: dict = {
        "run_id": "fmt-pers-r256",
        "cell_key": "fmt-pers",
        "behavior": "formatting",
        "rates_by_step": rates,
        "base_tier2": {"rate": base_rate},
    }
    if t2_rate is not None:
        e["tier2_trained"] = {"rate": t2_rate}
    return e


def test_reread_trigger_widened_fu5(fu5_round):
    runs = [r for r in fu4.FU5_RUNS if r.run_id == "fmt-pers-r256"]
    # mid-band case the WIDENED trigger exists for: rate 0.32 (delta +0.153
    # over base 0.167 — the legacy delta trigger would NOT fire)
    out = {"runs": {"fmt-pers-r256": _fmt_entry({"5": 0.32}, 1 / 6, None)}}
    assert fu4._formatting_reread_fires(out, runs) is True
    # Tier-2 confirmatory read alone also fires
    out = {"runs": {"fmt-pers-r256": _fmt_entry({"5": 0.10}, 1 / 6, 0.31)}}
    assert fu4._formatting_reread_fires(out, runs) is True
    # below the floor everywhere: no fire
    out = {"runs": {"fmt-pers-r256": _fmt_entry({"5": 0.29}, 1 / 6, 0.24)}}
    assert fu4._formatting_reread_fires(out, runs) is False


def test_reread_trigger_legacy_delta_fu4():
    runs = [r for r in fu4.FU4_RUNS if r.run_id == "fmt-pers-lr1e4"]
    entry = _fmt_entry({"5": 0.32}, 1 / 6, None)
    entry["run_id"] = "fmt-pers-lr1e4"
    # 0.32 - 0.167 = +0.153 < +0.30: the fu4 legacy delta trigger stays silent
    assert fu4._formatting_reread_fires({"runs": {"fmt-pers-lr1e4": entry}}, runs) is False
    entry2 = _fmt_entry({"5": 0.50}, 1 / 6, None)
    entry2["run_id"] = "fmt-pers-lr1e4"
    assert fu4._formatting_reread_fires({"runs": {"fmt-pers-lr1e4": entry2}}, runs) is True


@pytest.mark.parametrize(
    ("rate", "status"),
    [
        (0.0, "ok"),
        (0.35, "ok"),  # closed at the floor: degraded is STRICTLY above 0.35
        (0.36, "parity-degraded"),
        (0.50, "parity-degraded"),
        (0.51, "parity-failed"),
        (None, "missing"),
    ],
)
def test_retrain_parity_flag_fu5(fu5_round, rate, status):
    rates = {} if rate is None else {str(fu4.K3_PARITY_STEP): rate}
    out = {"runs": {"imp-bare-lr1e5": {"run_id": "imp-bare-lr1e5", "rates_by_step": rates}}}
    rec = fu4._retrain_parity_record(out)
    assert rec["status"] == status
    assert rec["abort_bar"] == fu4.K3_PARITY_MAX_RATE


def test_retrain_parity_fu4_has_no_degraded_band():
    out = {"runs": {"fmt-pers-lr1e5": {"run_id": "fmt-pers-lr1e5", "rates_by_step": {"15": 0.4}}}}
    rec = fu4._retrain_parity_record(out)
    assert rec["status"] == "ok"  # fu4: no degraded floor; abort bar unchanged


# ── A4 recipe-identity assert (committed fu4 manifest) ───────────────────────


def _committed_fu4_manifest_sha() -> str:
    path = FU4_DELIVERABLES / "cell_manifest_fu4.json"
    if not path.exists():
        pytest.skip("committed cell_manifest_fu4.json absent")
    man = json.loads(path.read_text())
    return {r["run_id"]: r for r in man["runs"]}["fmt-pers-lr1e4"]["train_mix_sha256"]


def test_a4_recipe_identity_assert(fu5_round):
    sha = _committed_fu4_manifest_sha()
    fu4._assert_reused_recipe_identity({"fmt-pers": {"train_mix_sha256": sha}})  # passes
    with pytest.raises(ValueError, match="A4 BROKEN"):
        fu4._assert_reused_recipe_identity({"fmt-pers": {"train_mix_sha256": "deadbeef"}})
    with pytest.raises(ValueError, match="fmt-pers cell not staged"):
        fu4._assert_reused_recipe_identity({})


# ── Eval-split diagnostic (fu5 D2 item 6) ────────────────────────────────────


def test_classify_eval_split_majority(fu5_round, tmp_path):
    """Majority over kept per-draw scores; zero kept draws -> unclassified
    (dropped, never coerced). The judge boundary is faked with a def that
    mirrors judge_graded's signature and returns a REAL JudgeResult."""
    questions = ["list three ways to save money", "write a short story", "weird one"]

    def fake_judge_graded(
        items,
        eval_prompt,
        *,
        n_draws,
        cache_dir,
        save_raw,
        judge_model,
        temperature=1.0,
        max_tokens=64,
        dry_run=False,
    ):
        assert n_draws == fu4.EVAL_SPLIT_N_DRAWS and max_tokens == fu4.JUDGE_MAX_TOKENS_FU4
        assert all(comp == fu4.EVAL_SPLIT_PLACEHOLDER for _i, _q, comp in items)
        per_item = {
            "evalsplit-q000": [90.0, 80.0, 20.0],  # 2/3 list draws -> list_affordable
            "evalsplit-q001": [10.0, 60.0, 20.0],  # 1/3 -> prose_natural
            "evalsplit-q002": [],  # all draws dropped -> unclassified
        }
        return JudgeResult(
            scores={k: (sum(v) / len(v) if v else None) for k, v in per_item.items()},
            n_total_draws=9,
            n_dropped_draws=3,
            per_item_scores=per_item,
        )

    cls = fu4._classify_eval_split(tmp_path, questions, judge_fn=fake_judge_graded)
    assert cls["evalsplit-q000"]["label"] == "list_affordable"
    assert cls["evalsplit-q001"]["label"] == "prose_natural"
    assert cls["evalsplit-q002"]["label"] == "unclassified"
    assert cls["evalsplit-q000"]["question"] == questions[0]


def test_eval_split_rates_real_predicate(fu5_round, tmp_path):
    """Per-split Tier-2 structural rates through the REAL formatting predicate
    (>=80% of non-empty answer lines are list items)."""
    listy = "- one\n- two\n- three"
    prose = "This is a flowing prose answer with no list structure at all."
    payload = {
        "questions": ["ql", "qp", "qx"],
        "completions": [[listy, prose], [prose, prose], [listy]],
    }
    f = tmp_path / "completions__trained__persona_software_engineer.json"
    f.write_text(json.dumps(payload))
    rec = fu4._eval_split_rates(
        f,
        {"ql": "list_affordable", "qp": "prose_natural"},  # qx unclassified
    )
    assert rec["list_affordable"]["n"] == 2 and rec["list_affordable"]["k"] == 1
    assert rec["prose_natural"]["n"] == 2 and rec["prose_natural"]["k"] == 0
    assert rec["n_completions_unclassified_questions"] == 1
    assert rec["list_affordable"]["rate"] == pytest.approx(0.5)
