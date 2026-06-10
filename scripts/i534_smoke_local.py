# ruff: noqa: RUF001, RUF003  # em-dash + Qwen marker " ※" + Greek ΔG intentional
#!/usr/bin/env python3
"""Task #534 — CPU-only end-to-end smoke for the local VM (no GPU).

Exercises every CPU-runnable portion of the #534 pipeline on REAL components
(the i504_smoke_local.py precedent). Phases:

  A. pool-fetch         — REAL HF download of the #530 train pool bytes via
                          the run-cell helper (the production fetch path).
  B. train+snapshot     — the REAL `_maybe_attach_marker_band_stop` wiring +
                          REAL `MarkerBandStopCallback` (snapshot cadence,
                          stop predicate read path, band_stop_meta sidecar)
                          driven by a REAL `transformers.Trainer.train()` on
                          a tiny random Qwen2 + the REAL Qwen tokenizer +
                          REAL pool rows. (The production 7B `train_lora`
                          load is GPU-bound: device_map + bf16.)
  C. fraction-selection — the REAL `scripts/i534_select_fractions.py` CLI on
                          phase B's snapshots, INCLUDING the per-step
                          source-trajectory PEFT-adapter loop on CPU
                          (uploads + deletion skipped via --hf-repo '').
  D. eval (z-schema)    — REAL `compute_kl_and_slot_stats_for_checkpoint` on
                          the tiny model: asserts finite KL + finite Δz
                          (plan §13.10). Plus the sweep dispatcher DRY-RUN
                          (subprocess shape + terminal [phase=done]).
  E. bystander          — REAL `i534_emit_bystander_resolution.py` over a
                          4-fraction fixture slab derived from the COMMITTED
                          #530 trajectories (real measured data).
  F. analysis           — REAL `i534_trajectory_analyze.py` over the fixture;
                          self-test: the fixture's frac=1.00 == #530's data,
                          so the replication check must report sign_match
                          with |Δρ| ≈ 0 against analysis_v1.json.
  G. figures            — REAL `issue534_make_figures.py` over the fixture.

Run from the worktree root:
    uv run python scripts/i534_smoke_local.py
"""

from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_CELL = "c504v3_near"
SMOKE_SEED = 42


def setup_log() -> logging.Logger:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [smoke=%(name)s] %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger("i534_smoke")


def _import_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_tiny_qwen(out_dir: Path) -> Path:
    """Tiny random Qwen2 with the REAL Qwen-2.5-7B-Instruct tokenizer.

    Full vocab (so marker id 83399 + eos 151645 resolve) but 2 layers /
    hidden 16 — CPU-trainable in seconds.
    """
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    assert tok.encode(" ※", add_special_tokens=False) == [83399]
    cfg = Qwen2Config(
        vocab_size=152064,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=4096,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(cfg)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def phase_a_pool_fetch(log, work: Path) -> Path:
    run_cell = _import_script("i534_run_cell")
    pool = work / "train_pool.jsonl"
    run_cell._fetch_train_pool_from_hf(SMOKE_CELL, SMOKE_SEED, pool)
    n_rows = sum(1 for line in pool.read_text().splitlines() if line.strip())
    assert n_rows > 0
    log.info("PHASE A PASS — pool-fetch: %s (%d rows)", pool, n_rows)
    return pool


def phase_b_train_with_snapshots(log, work: Path, tiny_dir: Path, pool: Path) -> Path:
    """REAL wiring fn + REAL callback + REAL Trainer.train() on the tiny model."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from explore_persona_space.train.sft import TrainLoraConfig, _maybe_attach_marker_band_stop

    tok = AutoTokenizer.from_pretrained(tiny_dir)
    model = AutoModelForCausalLM.from_pretrained(tiny_dir, torch_dtype=torch.float32)
    peft_model = get_peft_model(
        model,
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, target_modules=["q_proj", "v_proj"]),
    )

    # Tiny REAL-row dataset (first 4 pool rows), labels = input_ids.
    rows = [json.loads(line) for line in pool.read_text().splitlines() if line.strip()][:4]

    def _tok_row(row):
        ids = tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=True)[:256]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)}

    train_ds = [_tok_row(r) for r in rows]

    snap_dir = work / "snapshots"
    targs = TrainingArguments(
        output_dir=str(work / "trainer_out"),
        max_steps=6,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-4,
        logging_steps=1,
        save_strategy="no",
        report_to=[],  # WANDB_INTENTIONALLY_DISABLED: local CPU smoke harness, no run telemetry
        seed=0,
        use_cpu=True,
    )
    trainer = Trainer(model=peft_model, args=targs, train_dataset=train_ds)

    # The REAL wiring function builds the probe from the REAL pool rows and
    # attaches the REAL callback with the #534 snapshot config.
    cfg = TrainLoraConfig(
        marker_only_loss=True,
        marker_text=" ※",
        marker_band_eval_every_steps=1,
        marker_band_min_steps=2,
        marker_band_probe_max_rows=4,
        marker_band_snapshot_every_steps=1,
        marker_band_snapshot_dir=str(snap_dir),
        marker_band_snapshot_max_count=64,
    )
    n_before = len(trainer.callback_handler.callbacks)
    _maybe_attach_marker_band_stop(trainer, tok, cfg, str(pool))
    assert len(trainer.callback_handler.callbacks) == n_before + 1, "callback not attached"

    trainer.train()

    meta_path = snap_dir / "band_stop_meta.json"
    assert meta_path.exists(), "band_stop_meta.json missing after train"
    meta = json.loads(meta_path.read_text())
    snaps = sorted(p.name for p in snap_dir.glob("step_*"))
    assert meta["stop_step"] == 6 and len(snaps) == 6, (meta, snaps)
    assert meta["snapshot_steps"] == [1, 2, 3, 4, 5, 6]
    assert len(meta["eval_history"]) >= 1 and all(
        math.isfinite(e["delta_nats"]) for e in meta["eval_history"]
    )
    log.info(
        "PHASE B PASS — train+snapshot: %d snapshots, stop_step=%d, stop_reason=%s, %d eval reads",
        len(snaps),
        meta["stop_step"],
        meta["stop_reason"],
        len(meta["eval_history"]),
    )
    return snap_dir


def phase_c_selection(log, work: Path, tiny_dir: Path, pool: Path, snap_dir: Path) -> Path:
    out_index = work / "checkpoint_index.json"
    manifest = work / "fraction_manifest.json"
    source_traj = work / "source_steps_trajectory.json"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/i534_select_fractions.py",
        "--snapshot-dir",
        str(snap_dir),
        "--train-jsonl",
        str(pool),
        "--checkpoint-index-out",
        str(out_index),
        "--manifest-out",
        str(manifest),
        "--source-traj-out",
        str(source_traj),
        "--hf-repo",
        "",
        "--base-model",
        str(tiny_dir),
        "--device",
        "cpu",
        "--probe-max-rows",
        "4",
    ]
    subprocess.run(cmd, env={**os.environ}, check=True, cwd=REPO_ROOT)
    idx = json.loads(out_index.read_text())
    man = json.loads(manifest.read_text())
    straj = json.loads(source_traj.read_text())
    # stop_step=6 → targets {2 (1.5 rounds to 2), 3, 4 (4.5 rounds to 4), 6}.
    assert set(idx) == {"0.25", "0.50", "0.75", "1.00"}, idx
    assert idx["1.00"]["step"] == 6
    assert man["logit_readout_valid"] is True, man["gauge_results"]
    assert man["stopped"] is False and man["stop_reason"] == "epoch_ceiling"
    assert len(straj["steps"]) == 6
    assert all(math.isfinite(s["delta_g_mean"]) for s in straj["steps"])
    assert all(math.isfinite(s["delta_z_marker_mean"]) for s in straj["steps"])
    # Unselected snapshots NOT deleted (uploads skipped).
    assert len(list(snap_dir.glob("step_*"))) == 6
    log.info(
        "PHASE C PASS — selection: index=%s, gauge valid, source-traj %d steps (finite ΔG+Δz)",
        {k: v["step"] for k, v in idx.items()},
        len(straj["steps"]),
    )
    return out_index


def phase_d_eval_z_schema_and_dry_run(log, work: Path, tiny_dir: Path, snap_dir: Path) -> None:
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        compute_kl_and_slot_stats_for_checkpoint,
    )

    personas = {"probe_a": "You are a careful historian.", "probe_b": "You are a chef."}
    questions = ["What is your favorite tool?", "Describe your morning."]
    r_text = {p: {q: "A short natural answer." for q in questions} for p in personas}
    adapter = sorted(snap_dir.glob("step_*"))[0]
    kl, slot_stats = compute_kl_and_slot_stats_for_checkpoint(
        base_model=str(tiny_dir),
        adapter_path=str(adapter),
        r_by_persona_q=r_text,
        eval_personas=personas,
        eval_questions=questions,
        device="cpu",
    )
    for p in personas:
        for q in questions:
            assert math.isfinite(kl[p][q]), (p, q, kl[p][q])
            st = slot_stats[p][q]
            for k in (
                "z_marker_trained",
                "z_marker_base",
                "z_eos_trained",
                "z_eos_base",
                "logz_trained",
                "logz_base",
                "delta_z_marker",
                "delta_z_margin",
            ):
                assert math.isfinite(st[k]), (p, q, k, st[k])
            # log P(marker) = z_marker − logZ identity at the slot.
            assert (
                abs(st["logp_marker_hf_trained"] - (st["z_marker_trained"] - st["logz_trained"]))
                < 1e-4
            )
    log.info("PHASE D-1 PASS — z-schema: finite KL + finite Δz/Δz_margin on %d pairs", 4)

    dry_log_dir = work / "logs"
    dry_log_dir.mkdir(exist_ok=True)
    out = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/i534_sweep.py",
            "--dry-run",
            "--cells",
            SMOKE_CELL,
            "--seeds",
            str(SMOKE_SEED),
            "--n-gpus",
            "1",
            "--arm-to-n-json",
            "eval_results/issue_530/phase0_5_gates.json",
            "--slab-root",
            str(work / "dry_slab"),
            "--log-dir",
            str(dry_log_dir),
        ],
        env={**os.environ},
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert "[phase=done]" in out.stdout, out.stdout[-2000:]
    assert "i534_run_cell.py" in out.stdout
    log.info("PHASE D-2 PASS — sweep dry-run: subprocess shape + [phase=done] emitted")

    # Signature smoke on the GPU-bound entrypoints (dispatcher → trainer ABI).
    import inspect

    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        run_trajectory_eval,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.train_cell import (
        train_one_cell,
    )

    t_sig = set(inspect.signature(train_one_cell).parameters)
    assert {
        "marker_band_snapshot_every_steps",
        "marker_band_snapshot_dir",
        "marker_band_snapshot_max_count",
    } <= t_sig, t_sig
    e_sig = set(inspect.signature(run_trajectory_eval).parameters)
    assert {"checkpoint_specs", "max_lora_rank", "max_model_len", "compute_kl"} <= e_sig, e_sig
    log.info("PHASE D-3 PASS — signature smoke on train_one_cell + run_trajectory_eval")


def build_fixture_slab(log, work: Path, source_traj: Path | None) -> Path:
    """4-fraction fixture slab from the COMMITTED #530 trajectories."""
    src = REPO_ROOT / "eval_results/issue_530"
    slab = work / "fixture_slab"
    slab.mkdir()
    fracs = (0.25, 0.50, 0.75, 1.00)
    n_cells = 0
    for cell_dir in sorted(src.glob("c504v3_*_seed*")):
        traj = json.loads((cell_dir / "trajectory.json").read_text())
        base_ck = traj["checkpoints"][-1]
        traj["checkpoints"] = []
        for f in fracs:
            ck = json.loads(json.dumps(base_ck))  # deep copy
            ck["frac"] = f
            ck["step"] = max(1, round(f * 20))
            traj["checkpoints"].append(ck)
        dest = slab / cell_dir.name
        dest.mkdir()
        (dest / "trajectory.json").write_text(json.dumps(traj))
        (dest / "fraction_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": "i534_fraction_manifest_v1",
                    "manifest": [
                        {
                            "frac": f,
                            "target_step": max(1, round(f * 20)),
                            "selected_step": max(1, round(f * 20)),
                            "exact": True,
                        }
                        for f in fracs
                    ],
                    "distinct_steps": 4,
                    "stopped": True,
                    "stop_reason": "band",
                    "logit_readout_valid": True,
                }
            )
        )
        n_cells += 1
    if source_traj is not None and source_traj.exists():
        shutil.copyfile(source_traj, slab / "c504v3_near_seed42" / "source_steps_trajectory.json")
    log.info("fixture slab: %d cells × 4 fractions at %s", n_cells, slab)
    return slab


def phase_e_bystander(log, slab: Path) -> None:
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/i534_emit_bystander_resolution.py",
            "--slab-root",
            str(slab),
        ],
        env={**os.environ},
        check=True,
        cwd=REPO_ROOT,
    )
    files = sorted(slab.glob("c504v3_*_seed*/bystander_resolution.json"))
    assert len(files) == 10, len(files)
    one = json.loads(files[0].read_text())
    assert set(one["per_fraction"]) == {"0.25", "0.50", "0.75", "1.00"}
    assert one["per_fraction"]["1.00"]["de_saturation_gate"]["verdict"] in ("PASS", "FAIL")
    log.info("PHASE E PASS — bystander: 10 files, per_fraction has 4 entries each")


def phase_f_analysis(log, slab: Path) -> Path:
    out = slab / "analysis_per_fraction.json"
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/i534_trajectory_analyze.py",
            "--slab-root",
            str(slab),
            "--n-boot",
            "60",
            "--boot-seed",
            "7",
            "--out",
            str(out),
        ],
        env={**os.environ},
        check=True,
        cwd=REPO_ROOT,
    )
    d = json.loads(out.read_text())
    assert set(d["per_fraction"]) == {"0.25", "0.50", "0.75", "1.00"}
    rep = d["replication_check"]
    assert rep["available"] is True
    # Self-test: fixture frac=1.00 == #530's own data → exact sign match,
    # |Δρ| ≈ 0 against the committed analysis_v1.json.
    for p in ("shadow_angle", "d_nearest_neg_nd"):
        pp = rep["per_predictor"][p]
        assert pp["sign_match"] is True, pp
        assert pp["abs_delta_rho"] < 1e-9, pp["abs_delta_rho"]
        assert pp["within_tolerance"] is True
        assert pp["ci_534"]["lo"] is not None
    # Zero-variance training_step flag must fire (all cells share one step).
    assert all(d["per_fraction"][f]["zero_variance_training_step"] for f in d["per_fraction"])
    # z fields absent in the #530-derived fixture → logit column dropped path.
    assert d["per_fraction"]["1.00"]["z_agreement"]["available"] is False
    assert (slab / "analysis_frac1.00_banded.json").exists()
    log.info(
        "PHASE F PASS — analysis: 4 per-fraction fits; replication self-test "
        "sign_match + |Δρ|<1e-9 vs committed analysis_v1.json; "
        "zero-variance flag fired; z-absent path exercised"
    )
    return out


def phase_g_figures(log, slab: Path, work: Path) -> None:
    out_dir = work / "figures"
    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/issue534_make_figures.py",
            "--slab-root",
            str(slab),
            "--out-dir",
            str(out_dir),
        ],
        env={**os.environ},
        check=True,
        cwd=REPO_ROOT,
    )
    pngs = sorted(p.name for p in out_dir.glob("*.png"))
    assert len(pngs) >= 5, pngs
    log.info("PHASE G PASS — figures: %d PNGs (%s)", len(pngs), ", ".join(pngs))


def main() -> int:
    log = setup_log()
    work = Path(tempfile.mkdtemp(prefix="issue-534-smoke-"))
    log.info("work dir: %s", work)

    pool = phase_a_pool_fetch(log, work)
    tiny_dir = build_tiny_qwen(work / "tiny_qwen")
    snap_dir = phase_b_train_with_snapshots(log, work, tiny_dir, pool)
    phase_c_selection(log, work, tiny_dir, pool, snap_dir)
    phase_d_eval_z_schema_and_dry_run(log, work, tiny_dir, snap_dir)
    slab = build_fixture_slab(log, work, work / "source_steps_trajectory.json")
    phase_e_bystander(log, slab)
    phase_f_analysis(log, slab)
    phase_g_figures(log, slab, work)
    log.info("ALL PHASES PASS — artifacts under %s", work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
