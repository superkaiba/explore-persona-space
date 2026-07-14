"""CPU pins for the #1090 fu4 (extended-dose-lr) driver.

Covers the plan-v6 registered mechanics: the 9-run D1 matrix, the recipe
overrides at the spec seam, the {5..75} rung arithmetic, the K1 composition +
sha gates, the K2 divergence predicate (incl. the first-logged-loss floor —
the fix for the tiny-real smoke's ln(vocab)~11.9 false-diverged, which the
ABSOLUTE threshold flagged on every smoke run pre-fix), the degeneracy guard,
and the selected+final adapter retention (ruled-out rungs deleted only after
the kept uploads verify). Real bodies throughout; the only fake is the
external Hub upload boundary (signature-conformant recording fn).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue1090_fu4 as fu4  # noqa: E402
import issue1090_run as i1090  # noqa: E402


def test_run_matrix_matches_plan_d1():
    assert len(fu4.FU4_RUNS) == 9
    ids = sorted(r.run_id for r in fu4.FU4_RUNS)
    expected = sorted(
        f"{cell}-{tag}"
        for cell in ("fmt-pers", "imp-pers", "imp-conv")
        for tag in ("lr1e5", "lr3e5", "lr1e4")
    )
    assert ids == expected
    conv = fu4.RUN_BY_ID["imp-conv-lr1e5"]
    assert conv.mix_layout == "fu3-flat"
    assert conv.mix_hub_prefix == "issue1090_fu3/C2-conv-con-impolite-claude"
    assert conv.context_id == "wildchat_prefix_real545"
    parent = fu4.RUN_BY_ID["fmt-pers-lr1e4"]
    assert parent.mix_hub_prefix.endswith("c1-formatting-claude/mix")
    assert parent.lr == 1e-4


def test_recipe_spec_carries_fu4_deviations_only():
    spec = fu4.fu4_recipe_spec("impolite", 3e-5)
    ov = spec.overrides
    assert ov["lr"] == 3e-5
    assert ov["epochs"] == 15
    assert ov["save_steps"] == 5
    assert ov["max_length"] == 2048
    # Everything else inherited verbatim from UNIFIED_OVERRIDES.
    assert (ov["lora_r"], ov["lora_alpha"], ov["lora_dropout"]) == (32, 64, 0.05)
    assert (ov["batch_size"], ov["grad_accum"]) == (4, 4)
    assert ov["save_only_model"] is True


def test_expected_rungs_80_rows():
    rungs, total = fu4.fu4_expected_rungs(80)
    assert total == 75
    assert rungs == list(range(5, 76, 5))


def _write_state(tmp_path: Path, losses: list[float]) -> dict[int, Path]:
    ckpt = tmp_path / "checkpoint-75"
    ckpt.mkdir(parents=True)
    hist = [{"step": i + 1, "loss": ls} for i, ls in enumerate(losses)]
    (ckpt / "trainer_state.json").write_text(json.dumps({"log_history": hist}))
    return {75: ckpt}


def test_k2_flat_high_initial_loss_is_not_divergence(tmp_path):
    # The tiny-real smoke regression: random-init loss ~ln(vocab) ~= 11.9 flat.
    # Pre-fix (absolute 5.0 bar) this flagged diverged; the first-logged-loss
    # floor makes "elevated initial condition" distinct from degradation.
    out = fu4.check_divergence(_write_state(tmp_path, [11.9] * 8))
    assert out["diverged"] is False
    assert out["effective_bar"] == pytest.approx(11.9)


def test_k2_sustained_blowup_from_low_start_diverges(tmp_path):
    # Production shape: 7B SFT starts ~1.5, so the effective bar is EXACTLY
    # the registered 5.0; 5 consecutive logged losses above it flag.
    losses = [1.5, 2.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    out = fu4.check_divergence(_write_state(tmp_path, losses))
    assert out["diverged"] is True
    assert out["at_step"] == 7
    assert out["effective_bar"] == pytest.approx(5.0)


def test_k2_short_excursion_does_not_diverge(tmp_path):
    losses = [1.5, 6.0, 6.0, 6.0, 6.0, 1.4, 1.3, 1.2]  # only 4 consecutive
    assert fu4.check_divergence(_write_state(tmp_path, losses))["diverged"] is False


def test_k2_nan_diverges(tmp_path):
    out = fu4.check_divergence(_write_state(tmp_path, [1.5, float("nan"), 1.4]))
    assert out["diverged"] is True
    assert out["reason"] == "nan_loss"


def test_degeneracy_stats_flags():
    long_diverse = " ".join(f"tok{i}" for i in range(60))
    assert fu4.degeneracy_stats([[long_diverse]])["degenerate"] is False
    short = "just three tokens"
    assert fu4.degeneracy_stats([[short]])["degenerate"] is True
    repetitive = " ".join(["spam ham"] * 40)  # 4-gram repetition >> 0.5
    rec = fu4.degeneracy_stats([[repetitive + " " + long_diverse], [long_diverse]])
    assert rec["max_4gram_repeat_frac"] > fu4.DEGEN_MAX_REPEAT_FRAC
    assert rec["degenerate"] is True


def _mix_fixture(tmp_path: Path, run, counts: dict[str, int], behavior: str) -> Path:
    mix_dir = tmp_path / run.run_id / "mix"
    mix_dir.mkdir(parents=True)
    n = sum(counts.values())
    with open(mix_dir / "train_mix.jsonl", "w") as f:
        for i in range(n):
            f.write(json.dumps({"prompt": [], "completion": [], "i": i}) + "\n")
    (mix_dir / "mix_meta.json").write_text(
        json.dumps({"counts_realized": counts, "spec": {"behavior_name": behavior}})
    )
    return mix_dir


def _cfg(tmp_path: Path, run, smoke: bool = False) -> i1090.RunConfig:
    return i1090.RunConfig(smoke=smoke, cells=(run,), out_root=tmp_path, upload=False)


def test_verify_fu4_mix_composition_gate(tmp_path):
    run = fu4.RUN_BY_ID["imp-pers-lr1e5"]
    _mix_fixture(tmp_path, run, {"positives": 20, "negatives": 20, "generic": 40}, "impolite")
    rec = fu4.verify_fu4_mix(_cfg(tmp_path, run), run, None)
    assert rec["train_mix_sha256"]
    assert rec["hf_prefix"] == run.mix_hub_prefix
    assert rec["mix_layout"] == "parent-mix-subdir"


def test_verify_fu4_mix_rejects_wrong_composition(tmp_path):
    run = fu4.RUN_BY_ID["imp-pers-lr1e5"]
    _mix_fixture(tmp_path, run, {"positives": 30, "negatives": 10, "generic": 40}, "impolite")
    with pytest.raises(ValueError, match="composition"):
        fu4.verify_fu4_mix(_cfg(tmp_path, run), run, None)


def test_verify_fu4_mix_rejects_manifest_sha_drift(tmp_path):
    run = fu4.RUN_BY_ID["imp-pers-lr1e5"]
    _mix_fixture(tmp_path, run, {"positives": 20, "negatives": 20, "generic": 40}, "impolite")
    with pytest.raises(ValueError, match="manifest pin"):
        fu4.verify_fu4_mix(_cfg(tmp_path, run), run, "deadbeef" * 8)


def test_upload_keeps_selected_and_final_rungs_deletes_rest(tmp_path):
    """The plan-§9 retention policy: selected + final rungs upload; the
    ruled-out rungs (the §10 declared discard) are deleted ONLY after the kept
    uploads returned URLs. Real upload_fu4_run body; the recording fn fakes
    only the external Hub boundary (exact seam signature)."""
    run = fu4.RUN_BY_ID["fmt-pers-lr1e5"]
    adapter_root = tmp_path / run.run_id / "train"
    for step in (5, 10, 15, 20):
        (adapter_root / f"checkpoint-{step}").mkdir(parents=True)
        (adapter_root / f"checkpoint-{step}" / "adapter_model.safetensors").write_text("x")
    rec = {
        "status": "trained",
        "adapter_root": str(adapter_root),
        "selected_ckpt": str(adapter_root / "checkpoint-10"),
        "selection": {"step": 10, "rate": 0.7, "in_band": True, "fallback": None},
    }
    calls: list[str] = []

    def recording_upload(local_path, repo_id, repo_type, path_in_repo, **kw) -> str:
        calls.append(path_in_repo)
        return f"fake://{repo_id}/{path_in_repo}"

    cfg = _cfg(tmp_path, run)
    seams = i1090.Seams1090(upload_fn=recording_upload)
    fu4.upload_fu4_run(cfg, seams, run, rec)
    adapter_pirs = sorted(p for p in calls if p.startswith("adapters/"))
    assert adapter_pirs == [
        f"adapters/issue1090_fu4/{run.run_id}/checkpoint-10",
        f"adapters/issue1090_fu4/{run.run_id}/checkpoint-20",
    ]
    kept = sorted(p.name for p in adapter_root.glob("checkpoint-*"))
    assert kept == ["checkpoint-10", "checkpoint-20"]


def test_upload_failure_aborts_before_any_rung_delete(tmp_path):
    run = fu4.RUN_BY_ID["fmt-pers-lr1e5"]
    adapter_root = tmp_path / run.run_id / "train"
    for step in (5, 10):
        (adapter_root / f"checkpoint-{step}").mkdir(parents=True)
    rec = {
        "status": "trained",
        "adapter_root": str(adapter_root),
        "selected_ckpt": str(adapter_root / "checkpoint-5"),
        "selection": {"step": 5, "rate": 0.7, "in_band": True, "fallback": None},
    }

    def empty_upload(local_path, repo_id, repo_type, path_in_repo, **kw) -> str:
        return ""  # the fail-loud empty-return contract

    with pytest.raises(RuntimeError, match="refusing silent loss"):
        fu4.upload_fu4_run(_cfg(tmp_path, run), i1090.Seams1090(upload_fn=empty_upload), run, rec)
    assert sorted(p.name for p in adapter_root.glob("checkpoint-*")) == [
        "checkpoint-10",
        "checkpoint-5",
    ]
