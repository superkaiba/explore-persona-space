"""#1112 — accelerate/DeepSpeed launch composition pins (config + world size + CVD).

Round-2 Critical 4: transformers' ``HfTrainerDeepSpeedConfig.trainer_config_process``
``fill_match`` RAISES ValueError at Trainer init when the DS config's explicit
(non-"auto") ``gradient_accumulation_steps`` differs from the TrainingArguments
value. The m2 marker trainer pins accum 16 (the #514 eff-batch-64 recipe) and
was launched against the accum-1 yaml — a guaranteed pod-side crash. This test
pins every (launch config, trainer accum) pair AND that the dispatcher wires m2
to the accum-16 config.

Round-4 crash (pod-1112, 4xH100): s2's IN-PROCESS train_lora call set
CUDA_VISIBLE_DEVICES=0 in the dispatcher's own env (the train/sft.py clobber),
so s3's full-FT compose read 1 visible GPU -> ``--num_processes 1`` against the
4-GPU ZeRO-3 config (zero sharding, whole-7B OOM on GPU 0) AND the subprocess
inherited CVD=0. The tests below pin the fixed composition parse-level (no GPU):
full mode = whole-pod ``--num_processes 4`` + explicit 4-GPU CVD env regardless
of the clobber; smoke mode = single-process (the proven 1-GPU GCE smoke shape).
"""

from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

FOUR_GPUS = ["0", "1", "2", "3"]


def _ds_accum(config_rel: str) -> int | str:
    cfg = yaml.safe_load((REPO_ROOT / config_rel).read_text())
    return cfg["deepspeed_config"]["gradient_accumulation_steps"]


def test_behavior_ft_accum_matches_accum1_config():
    import issue1112_dispatch as d

    from explore_persona_space.experiments import issue_1112 as C

    accum = _ds_accum(d.ACCEL_CONFIG)
    # the dispatcher passes --grad-accum C.FT_GRAD_ACCUM to train_behavior_fullft
    assert accum in ("auto", C.FT_GRAD_ACCUM), (accum, C.FT_GRAD_ACCUM)


def test_marker_ft_accum_matches_accum16_config():
    import issue1112_dispatch as d
    import issue1112_train_marker_fullft as trainer

    accum = _ds_accum(d.MARKER_ACCEL_CONFIG)
    assert accum in ("auto", trainer.FT_GRAD_ACCUM), (accum, trainer.FT_GRAD_ACCUM)
    # eff-batch 64 recipe intact (#514 ft_b1: per-device 1 x accum 16 x 4 GPUs)
    assert trainer.FT_BATCH_SIZE_PER_DEVICE * trainer.FT_GRAD_ACCUM * 4 == 64


def test_dispatcher_wires_m2_to_marker_accel_config(monkeypatch, tmp_path):
    """The m2 launch uses MARKER_ACCEL_CONFIG (not the behavior-FT config) —
    parse-level on the composed command (round 4 extracted it to _marker_ft_cmd)."""
    import issue1112_dispatch as d

    assert d.MARKER_ACCEL_CONFIG != d.ACCEL_CONFIG
    src = inspect.getsource(d.phase_train)
    m2_branch = src.split('cell == "m2_fullft_band8"', 1)[1].split("else:", 1)[0]
    assert "_marker_ft_cmd" in m2_branch
    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: list(FOUR_GPUS))
    cfg = d.Cfg(smoke=False, cells=("m2_fullft_band8",), out_root=tmp_path)
    cmd = d._marker_ft_cmd(cfg, "m2_fullft_band8", out_dir=tmp_path / "t", grid=(2, 4))
    assert cmd[cmd.index("--config_file") + 1] == d.MARKER_ACCEL_CONFIG
    assert cmd[cmd.index("--num_processes") + 1] == "4"
    assert cmd[cmd.index("--max-steps") + 1] == "4"
    assert (REPO_ROOT / d.MARKER_ACCEL_CONFIG).exists()


# ── Round-4 launch-composition pins (num_processes + CVD, full vs smoke) ─────


def test_ft_num_processes_pinned_to_both_accel_configs():
    """FT_NUM_PROCESSES == the yamls' num_processes (eff-batch contract)."""
    import issue1112_dispatch as d

    for rel in (d.ACCEL_CONFIG, d.MARKER_ACCEL_CONFIG):
        cfgy = yaml.safe_load((REPO_ROOT / rel).read_text())
        assert cfgy["num_processes"] == d.FT_NUM_PROCESSES, rel


def test_physical_gpu_ids_honors_launcher_cvd(monkeypatch):
    """A LAUNCHER-set CVD (captured at import) is a deliberate restriction."""
    import issue1112_dispatch as d

    monkeypatch.setattr(d, "_INITIAL_CVD", "4,5")
    assert d._physical_gpu_ids() == ["4", "5"]


def test_physical_gpu_ids_enumerates_via_nvidia_smi(monkeypatch):
    """Real body: no launcher CVD -> subprocess nvidia-smi enumeration
    (clobber-immune — never torch.cuda.device_count)."""
    import issue1112_dispatch as d

    monkeypatch.setattr(d, "_INITIAL_CVD", None)
    fake_run = mock.create_autospec(subprocess.run)
    fake_run.return_value = subprocess.CompletedProcess(
        args=["nvidia-smi"], returncode=0, stdout="0\n1\n2\n3\n", stderr=""
    )
    monkeypatch.setattr(d.subprocess, "run", fake_run)
    assert d._physical_gpu_ids() == FOUR_GPUS
    assert fake_run.call_args.args[0][0] == "nvidia-smi"


def test_physical_gpu_ids_fails_loud_without_gpus(monkeypatch):
    import issue1112_dispatch as d

    monkeypatch.setattr(d, "_INITIAL_CVD", None)

    def _raise(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(d.subprocess, "run", _raise)
    with pytest.raises(RuntimeError, match="no CUDA devices"):
        d._physical_gpu_ids()


def test_ft_cmd_full_mode_composes_whole_pod_launch(monkeypatch, tmp_path):
    import issue1112_dispatch as d

    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: list(FOUR_GPUS))
    cfg = d.Cfg(smoke=False, cells=("s3_fullft_neg",), out_root=tmp_path)
    cmd = d._ft_cmd(cfg, "s3_fullft_neg", out_dir=tmp_path / "t", max_steps=30, ckpt_steps=(2, 4))
    assert cmd[cmd.index("--num_processes") + 1] == "4", cmd
    assert cmd[cmd.index("--config_file") + 1] == d.ACCEL_CONFIG


def test_ft_cmd_smoke_mode_single_process(tmp_path):
    """Smoke FT stays single-process BY DESIGN (runs on 1-GPU smoke instances;
    the mode gate never consults the physical-GPU count in smoke)."""
    import issue1112_dispatch as d

    cfg = d.Cfg(smoke=True, cells=("s3_fullft_neg",), out_root=tmp_path)
    cmd = d._ft_cmd(cfg, "s3_fullft_neg", out_dir=tmp_path / "t", max_steps=2, ckpt_steps=(2,))
    assert cmd[cmd.index("--num_processes") + 1] == "1", cmd


def test_ft_num_processes_full_fails_loud_below_world_size(monkeypatch, tmp_path):
    """Full mode NEVER degrades the ZeRO-3 world size — it fails loud."""
    import issue1112_dispatch as d

    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: ["0"])
    cfg = d.Cfg(smoke=False, cells=("s3_fullft_neg",), out_root=tmp_path)
    with pytest.raises(RuntimeError, match="only 1 physical GPU"):
        d._ft_num_processes(cfg)


def test_ft_env_resets_inprocess_cvd_clobber(monkeypatch, tmp_path):
    """The FT subprocess env carries an EXPLICIT 4-GPU CVD even when the
    dispatcher env holds the train_lora clobber (CUDA_VISIBLE_DEVICES=0)."""
    import issue1112_dispatch as d

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")  # the in-process clobber
    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: list(FOUR_GPUS))
    cfg = d.Cfg(smoke=False, cells=("s3_fullft_neg",), out_root=tmp_path)
    assert d._ft_env(cfg)["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_phase_train_ft_after_inprocess_lora_composes_whole_pod_launch(monkeypatch, tmp_path):
    """Round-4 regression pin — the exact crash shape, tiny-real on CPU.

    s2 trains in-process and clobbers the dispatcher env to CVD=0 (as
    train/sft.py really does); s3's full-FT launch must STILL compose
    ``--num_processes 4`` with an explicit 0,1,2,3 CVD env. Pre-fix this
    fails: ``min(4, torch.cuda.device_count())`` read the clobbered env and
    the subprocess inherited CVD=0. Executes the REAL phase_train body,
    _ft_cmd, _run_ft_subprocess, _ft_env, and _fresh_ft_out_dir; fakes only
    the GPU boundaries (_train_lora_cell / _run_subprocess), signature-mirrored.
    """
    import issue1112_dispatch as d

    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: list(FOUR_GPUS))
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_train_lora_cell(cfg, cell, train_cfg):
        # replicate the train/sft.py in-process clobber the real call performs
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        root = cfg.out_root / cell / "train"
        root.mkdir(parents=True, exist_ok=True)
        return {"adapter_root": str(root), "training_loss": 0.0}

    def fake_run_subprocess(cmd, log_path, env=None):
        calls.append((list(cmd), dict(env) if env is not None else None))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(d, "_train_lora_cell", fake_train_lora_cell)
    monkeypatch.setattr(d, "_run_subprocess", fake_run_subprocess)
    cfg = d.Cfg(smoke=False, cells=("s2_lora_pos", "s3_fullft_neg"), out_root=tmp_path)
    results = d.phase_train(cfg)

    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "0"  # the clobber DID happen
    assert set(results) == {"s2_lora_pos", "s3_fullft_neg"}
    assert len(calls) == 1  # exactly ONE subprocess: the s3 FT (s2 was in-process)
    ft_cmd, ft_env = calls[0]
    assert ft_cmd[ft_cmd.index("--num_processes") + 1] == "4", ft_cmd
    assert ft_env is not None and ft_env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"
    assert ft_cmd[ft_cmd.index("--config_file") + 1] == d.ACCEL_CONFIG
    assert (tmp_path / "s3_fullft_neg" / "build_result.json").exists()


def test_phase_train_clears_stale_partial_ft_out_dir(monkeypatch, tmp_path):
    """Element-5 disposition (wipe, driver-executed): a crashed FT cell's
    partial train dir (build_result.json ABSENT) is cleared before the fresh
    launch — the trainer never resumes (save_only_model=True) and stale
    checkpoint-* dirs would poison _enumerate_rungs."""
    import issue1112_dispatch as d

    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: list(FOUR_GPUS))
    monkeypatch.setattr(d, "_run_subprocess", lambda cmd, log_path, env=None: None)
    stale = tmp_path / "s3_fullft_neg" / "train" / "checkpoint-2"
    stale.mkdir(parents=True)
    (stale / "junk.bin").write_text("stale partial state")
    cfg = d.Cfg(smoke=False, cells=("s3_fullft_neg",), out_root=tmp_path)
    d.phase_train(cfg)
    assert not stale.exists()
    assert (tmp_path / "s3_fullft_neg" / "build_result.json").exists()


# ── Round-5 concern fix: g1-ext done-sentinel (train_metadata.json) ──────────


def _g1_ext_setup(d, tmp_path, monkeypatch):
    """Drive the REAL phase_g1_gate body to the extension loop for one cell.

    g1 fires (all rates below C.INSTALL_FLOOR), the fence pre-check resolves
    extend_in_place (fence_hours=72 vs ~28h default projection), and only the
    GPU boundary (_physical_gpu_ids / _run_subprocess) is faked."""
    monkeypatch.setattr(d, "_physical_gpu_ids", lambda: list(FOUR_GPUS))
    calls: list[tuple[list[str], dict[str, str] | None]] = []

    def fake_run_subprocess(cmd, log_path, env=None):
        calls.append((list(cmd), dict(env) if env is not None else None))
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(d, "_run_subprocess", fake_run_subprocess)
    cell_root = tmp_path / "s3_fullft_neg"
    cell_root.mkdir(parents=True)
    # the original s3 training's dispatcher-written sentinel already exists at
    # cell_root — exactly why it cannot double as the ext done-sentinel
    (cell_root / "build_result.json").write_text(
        json.dumps({"adapter_root": str(cell_root / "train")})
    )
    cfg = d.Cfg(smoke=False, cells=("s3_fullft_neg",), out_root=tmp_path, fence_hours=72.0)
    selections = {"s3_fullft_neg": {"rates_by_step": {"30": 0.10}}}  # < INSTALL_FLOOR
    return cfg, selections, cell_root, calls


def test_g1_ext_completed_dir_never_wiped_or_retrained(monkeypatch, tmp_path):
    """Concern g1-ext-done-sentinel-never-written (code-review v4): the guard
    keys on the TRAINER-written train_metadata.json — the only root-level file
    train_behavior_fullft.py produces (save_only_model=True; no root
    config.json). A COMPLETED extension on resume is classified done: no wipe,
    no retrain. Fails pre-fix (config.json key -> completed tree wiped +
    retrained)."""
    import issue1112_dispatch as d

    cfg, selections, cell_root, calls = _g1_ext_setup(d, tmp_path, monkeypatch)
    ext_dir = cell_root / "train_ext"
    ckpt = ext_dir / "checkpoint-60"
    ckpt.mkdir(parents=True)
    (ckpt / "model.safetensors").write_text("completed weights")
    (ext_dir / "train_metadata.json").write_text(json.dumps({"saved_checkpoints": [60]}))

    rec = d.phase_g1_gate(cfg, selections)

    assert rec["action"] == "extend_in_place"
    assert calls == []  # completed -> never retrained
    assert (ckpt / "model.safetensors").read_text() == "completed weights"  # never wiped
    build = json.loads((cell_root / "build_result.json").read_text())
    assert build["g1_extension"] is True and build["adapter_root"] == str(ext_dir)


def test_g1_ext_partial_dir_wiped_and_retrained(monkeypatch, tmp_path):
    """Partial ext (no train_metadata.json) keeps the round-4 disposition:
    stale checkpoint-* wiped before ONE fresh whole-pod extension launch."""
    import issue1112_dispatch as d

    from explore_persona_space.experiments import issue_1112 as C

    cfg, selections, cell_root, calls = _g1_ext_setup(d, tmp_path, monkeypatch)
    stale = cell_root / "train_ext" / "checkpoint-2"
    stale.mkdir(parents=True)
    (stale / "junk.bin").write_text("stale partial state")

    rec = d.phase_g1_gate(cfg, selections)

    assert rec["action"] == "extend_in_place"
    assert not stale.exists()  # partial -> wiped before relaunch
    assert len(calls) == 1
    cmd, env = calls[0]
    assert cmd[cmd.index("--max-steps") + 1] == str(C.G1_EXTENSION_STEP_CEILING)
    assert env is not None and env["CUDA_VISIBLE_DEVICES"] == "0,1,2,3"


def test_run_subprocess_real_body(tmp_path):
    """Real _run_subprocess body: env passthrough + log write + fail-loud rc."""
    import issue1112_dispatch as d

    log = tmp_path / "x.log"
    d._run_subprocess(["echo", "hi"], log, env={**os.environ, "EPM_TEST_VAR": "1"})
    assert "hi" in log.read_text()
    with pytest.raises(RuntimeError, match="rc="):
        d._run_subprocess(["false"], log)
