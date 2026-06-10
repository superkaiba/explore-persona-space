# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + Qwen marker " ※" + Greek ΔG intentional
#!/usr/bin/env python3
"""Task #555 — CPU-only end-to-end smoke for the local VM (no GPU).

Forked from scripts/i534_smoke_local.py. Exercises every CPU-runnable portion
of the #555 pipeline on REAL components. Phases:

  A. pool-build         — REAL `build_cell_504(seed=7)` (the production
                          fresh-seed rebuild path, NOT the retired HF fetch)
                          at full N; asserts the 400-row horizon invariant
                          via the worker's own `_assert_pool_rows`.
  B. train+hard-stop    — REAL `_maybe_attach_marker_band_stop` wiring + REAL
                          `MarkerBandStopCallback` (production-shaped
                          eval_every=10/min_steps=20 → inert below the stop)
                          + REAL `HardStopAtStepCallback(stop_at_step=5)`
                          driven by a REAL `transformers.Trainer.train()` on
                          a tiny random Qwen2 + the REAL Qwen tokenizer +
                          REAL pool rows. Asserts: stopped at global_step 5;
                          band_stop_meta stop_step==5 / stopped==false;
                          snapshots step_0001..step_0005; PLUS the negative
                          horizon test (expect_max_steps mismatch raises).
  C. fraction-selection — the REAL `scripts/i534_select_fractions.py` CLI
                          (reused verbatim by the #555 worker) with
                          `--fractions 1.0` on phase B's snapshots: frac 1.00
                          → step 5 EXACT; manifest carries
                          source_delta_g_at_selected_steps. Then the #555
                          worker's REAL `_assert_adapter_distinct` on the
                          selected snapshot (lora_B non-zero after 5 steps).
  D. eval (z-schema)    — REAL `compute_kl_for_checkpoint` (the merged main
                          implementation) + `assert_logit_readout_gauge_free`
                          on the tiny model: finite KL + four-floats fields +
                          the logp = z_marker − logZ identity. Plus the #555
                          sweep dispatcher DRY-RUN (single-cell smoke shape
                          AND the full 40-cell grid enumeration + terminal
                          [phase=done]).
  E. analysis           — REAL `i555_replicate_analyze.py` over a 3-replicate
                          fixture slab derived from the COMMITTED #534
                          trajectories (real measured producer output; the
                          cross-phase data-contract smoke), including a
                          `_bandctrl` decoy dir that MUST be excluded.
  F. figures            — REAL `i555_make_figures.py` over the fixture's
                          analysis_replicates.json.

Run from the worktree root:
    uv run python scripts/i555_smoke_local.py
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
SMOKE_SEED = 7
HARD_STOP = 5
TINY_MAX_STEPS = 25  # >= the band-stop's min_steps=20 so the production
# inertness shape holds (no _disabled_too_short branch).


def setup_log() -> logging.Logger:
    logging.basicConfig(
        level="INFO",
        format="%(asctime)s [smoke=%(name)s] %(levelname)s | %(message)s",
        stream=sys.stdout,
    )
    return logging.getLogger("i555_smoke")


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


def phase_a_pool_build(log, work: Path) -> Path:
    """REAL fresh-seed pool rebuild at full N (the production data-gen path)."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472.r_generate import (
        get_train_eval_questions,
        load_r_artifact,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504 import (
        MARKER_TEXT,
        SOURCE_PERSONA,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_504.build_training_data import (
        build_cell_504,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_530.data_deps import (
        prepare_data_dependencies,
    )

    prepare_data_dependencies()
    arm_to_n_payload = json.loads(
        (REPO_ROOT / "eval_results/issue_530/phase0_5_gates.json").read_text()
    )
    bank = load_persona_bank(REPO_ROOT / "data/issue_472/persona_bank.json")
    r_train = load_r_artifact(REPO_ROOT / "data/issue_472/on_policy_R/R_train.json")
    q_train, _q_eval = get_train_eval_questions()

    pool = work / "train_pool.jsonl"
    build_cell_504(
        SMOKE_CELL,
        pool,
        r_train=r_train,
        arm_to_positioned_n=arm_to_n_payload.get("arm_to_positioned_n", {}),
        q_train=q_train,
        persona_bank=bank,
        source=SOURCE_PERSONA,
        marker_text=MARKER_TEXT,
        smoke_mid_band_n=arm_to_n_payload.get("smoke_mid_band_n"),
        seed=SMOKE_SEED,
    )
    run_cell = _import_script("i555_run_cell")
    n_rows = run_cell._assert_pool_rows(pool)  # the worker's own 400-row assert
    log.info("PHASE A PASS — pool-build (seed=%d): %s (%d rows)", SMOKE_SEED, pool, n_rows)
    return pool


def phase_b_train_with_hard_stop(log, work: Path, tiny_dir: Path, pool: Path) -> Path:
    """REAL wiring + REAL band-stop + REAL HardStopAtStepCallback on the tiny model."""
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    from explore_persona_space.eval.callbacks import HardStopAtStepCallback
    from explore_persona_space.train.sft import TrainLoraConfig, _maybe_attach_marker_band_stop

    tok = AutoTokenizer.from_pretrained(tiny_dir)
    model = AutoModelForCausalLM.from_pretrained(tiny_dir, torch_dtype=torch.float32)
    peft_model = get_peft_model(
        model,
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, target_modules=["q_proj", "v_proj"]),
    )

    rows = [json.loads(line) for line in pool.read_text().splitlines() if line.strip()][:4]

    def _tok_row(row):
        ids = tok.apply_chat_template(row["prompt"] + row["completion"], tokenize=True)[:256]
        return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": list(ids)}

    train_ds = [_tok_row(r) for r in rows]

    def _make_trainer(max_steps: int):
        targs = TrainingArguments(
            output_dir=str(work / "trainer_out"),
            max_steps=max_steps,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-4,
            logging_steps=1,
            save_strategy="no",
            report_to=[],  # WANDB_INTENTIONALLY_DISABLED: local CPU smoke harness
            seed=0,
            use_cpu=True,
        )
        return Trainer(model=peft_model, args=targs, train_dataset=train_ds)

    # ── Negative horizon test FIRST: expect_max_steps mismatch must raise. ──
    trainer_bad = _make_trainer(TINY_MAX_STEPS)
    trainer_bad.add_callback(
        HardStopAtStepCallback(stop_at_step=HARD_STOP, expect_max_steps=TINY_MAX_STEPS + 1)
    )
    raised = False
    try:
        trainer_bad.train()
    except RuntimeError as e:
        raised = "scheduler horizon" in str(e)
    assert raised, "horizon-mismatch negative test did NOT raise"
    log.info("PHASE B(neg) PASS — horizon mismatch raised at train begin")

    # Re-init the adapter so the real run starts from lora_B == 0 again.
    peft_model = get_peft_model(
        AutoModelForCausalLM.from_pretrained(tiny_dir, torch_dtype=torch.float32),
        LoraConfig(task_type=TaskType.CAUSAL_LM, r=4, target_modules=["q_proj", "v_proj"]),
    )

    snap_dir = work / "snapshots"
    trainer = _make_trainer(TINY_MAX_STEPS)
    # PRODUCTION-shaped band-stop config: eval_every=10 / min_steps=20 →
    # provably inert below the step-5 hard stop (empty eval_history expected).
    cfg = TrainLoraConfig(
        marker_only_loss=True,
        marker_text=" ※",
        marker_band_eval_every_steps=10,
        marker_band_min_steps=20,
        marker_band_probe_max_rows=4,
        marker_band_snapshot_every_steps=1,
        marker_band_snapshot_dir=str(snap_dir),
        marker_band_snapshot_max_count=64,
    )
    n_before = len(trainer.callback_handler.callbacks)
    _maybe_attach_marker_band_stop(trainer, tok, cfg, str(pool))
    assert len(trainer.callback_handler.callbacks) == n_before + 1, "band-stop not attached"
    trainer.add_callback(
        HardStopAtStepCallback(stop_at_step=HARD_STOP, expect_max_steps=TINY_MAX_STEPS)
    )

    trainer.train()
    assert trainer.state.global_step == HARD_STOP, trainer.state.global_step
    assert trainer.state.max_steps == TINY_MAX_STEPS, trainer.state.max_steps

    # The #555 worker's REAL post-train asserts (stop_step / stopped / snapshots).
    run_cell = _import_script("i555_run_cell")
    run_cell._assert_hard_stop_artifacts(SMOKE_CELL, snap_dir, HARD_STOP)
    meta = json.loads((snap_dir / "band_stop_meta.json").read_text())
    assert meta["eval_history"] == [], (
        "band-stop eval fired below the hard stop — it must be inert "
        f"(eval_every=10, min_steps=20): {meta['eval_history']}"
    )
    snaps = sorted(p.name for p in snap_dir.glob("step_*"))
    log.info(
        "PHASE B PASS — train+hard-stop: stopped at step %d (max_steps=%d untouched), "
        "%d snapshots %s, stop_reason=%s (cosmetic, not asserted)",
        trainer.state.global_step,
        trainer.state.max_steps,
        len(snaps),
        snaps,
        meta["stop_reason"],
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
        "--fractions",
        "1.0",
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
    # frac 1.00 of the realized stop (5) → step 5 EXACT (plan §4.2).
    assert set(idx) == {"1.00"}, idx
    assert idx["1.00"]["step"] == HARD_STOP, idx
    assert man["manifest"][0]["exact"] is True, man["manifest"]
    assert man["logit_readout_valid"] is True, man["gauge_results"]
    assert man["stopped"] is False
    assert man["source_delta_g_at_selected_steps"]["1.00"] is not None, man
    assert len(straj["steps"]) == HARD_STOP
    assert all(math.isfinite(s["delta_g_mean"]) for s in straj["steps"])
    # ── The #555 worker's REAL adapter-distinctness guard on the selection. ──
    run_cell = _import_script("i555_run_cell")
    run_cell._assert_adapter_distinct(SMOKE_CELL, Path(idx["1.00"]["path"]))
    log.info(
        "PHASE C PASS — selection: frac 1.00 → step %d exact, gauge valid, "
        "source ΔG@selected=%.4f, adapter-distinctness PASS",
        idx["1.00"]["step"],
        man["source_delta_g_at_selected_steps"]["1.00"],
    )
    return out_index


def phase_d_eval_z_schema_and_dry_run(log, work: Path, tiny_dir: Path, snap_dir: Path) -> None:
    from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
        assert_logit_readout_gauge_free,
        compute_kl_for_checkpoint,
    )

    personas = {"probe_a": "You are a careful historian.", "probe_b": "You are a chef."}
    questions = ["What is your favorite tool?", "Describe your morning."]
    r_text = {p: {q: "A short natural answer." for q in questions} for p in personas}
    adapter = sorted(snap_dir.glob("step_*"))[0]
    # The gauge assert the production eval runs BEFORE scoring raw logits.
    assert_logit_readout_gauge_free(str(adapter))
    stats = compute_kl_for_checkpoint(
        base_model=str(tiny_dir),
        adapter_path=str(adapter),
        r_by_persona_q=r_text,
        eval_personas=personas,
        eval_questions=questions,
        device="cpu",
    )
    for p in personas:
        for q in questions:
            st = stats[p][q]
            assert math.isfinite(st["kl"]), (p, q, st["kl"])
            for k in (
                "z_marker_g",
                "z_marker_b",
                "z_eos_g",
                "z_eos_b",
                "logZ_g",
                "logZ_b",
                "logp_hf_g",
                "logp_hf_b",
            ):
                assert math.isfinite(st[k]), (p, q, k, st[k])
            # log P(marker) = z_marker − logZ identity at the slot.
            assert abs(st["logp_hf_g"] - (st["z_marker_g"] - st["logZ_g"])) < 1e-4
    log.info("PHASE D-1 PASS — z-schema: finite KL + four floats per side on 4 pairs")

    # ── Sweep dispatcher dry-runs (subprocess shape + terminal [phase=done]). ─
    dry_slab = work / "dry_slab"
    base_cmd = [
        "uv",
        "run",
        "python",
        "scripts/i555_sweep.py",
        "--dry-run",
        "--arm-to-n-json",
        "eval_results/issue_530/phase0_5_gates.json",
        "--slab-root",
        str(dry_slab),
        "--log-dir",
        str(work / "logs"),
    ]
    # (a) the smoke shape: sweep with one cell (PASS_UNIFIED definition).
    out = subprocess.run(
        [*base_cmd, "--n-gpus", "1", "--cells", SMOKE_CELL, "--seeds", str(SMOKE_SEED)],
        env={**os.environ},
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert "[phase=done]" in out.stdout, out.stdout[-2000:]
    assert "1 (cell, seed) pairs to run" in out.stdout, out.stdout[-2000:]
    # (b) the full production grid: 4 arms × 10 seeds = 40 cells enumerated.
    out = subprocess.run(
        [*base_cmd, "--n-gpus", "4"],
        env={**os.environ},
        check=True,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert "[phase=done]" in out.stdout, out.stdout[-2000:]
    assert "40 (cell, seed) pairs to run" in out.stdout, out.stdout[-2000:]
    dispatch = json.loads((dry_slab / "sweep_dispatch.json").read_text())
    assert len(dispatch) == 40, len(dispatch)
    assert all("i555_run_cell.py" in " ".join(d["cmd"]) for d in dispatch)
    assert all("--hard-stop-at-step" in d["cmd"] for d in dispatch)
    log.info(
        "PHASE D-2 PASS — dispatcher dry-runs: smoke shape (1 cell) + full grid "
        "(40 cells, hard-stop forwarded), terminal [phase=done] emitted"
    )


def build_fixture_slab(log, work: Path) -> Path:
    """3-replicate fixture slab from the COMMITTED #534 trajectories (real data).

    Replicate 1 = the parent's real seeds {42, 137}; replicates 2 and 3 are
    copies under the fresh-seed names {7, 11} / {19, 23} (same real producer
    bytes — the cross-phase data-contract smoke needs the real SHAPE, and the
    duplicated replicates make the cross-replicate t-interval + verdict path
    executable at n=3). Also plants a `_bandctrl` decoy dir that the analyzer
    MUST exclude (consistency-checker note 1).
    """
    src_slab = REPO_ROOT / "eval_results" / "issue_534"
    slab = work / "fixture_slab"
    slab.mkdir(parents=True, exist_ok=True)
    arms = ("c504v3_near", "c504v3_mid_near", "c504v3_mid_far", "c504v3_far")
    seed_map = {42: (42, 7, 19), 137: (137, 11, 23)}
    n_dirs = 0
    for arm in arms:
        for src_seed, dst_seeds in seed_map.items():
            src = src_slab / f"{arm}_seed{src_seed}"
            assert src.exists(), src
            for dst_seed in dst_seeds:
                dst = slab / f"{arm}_seed{dst_seed}"
                if not dst.exists():
                    shutil.copytree(src, dst)
                n_dirs += 1
    # The _bandctrl decoy (must be EXCLUDED by collect_manifest_flags_555).
    decoy = slab / "c504v3_near_seed7_bandctrl"
    if not decoy.exists():
        shutil.copytree(src_slab / "c504v3_near_seed42", decoy)
    log.info("fixture slab: %d production dirs + 1 _bandctrl decoy at %s", n_dirs, slab)
    return slab


def phase_e_analysis(log, slab: Path) -> Path:
    out = slab / "analysis_replicates.json"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/i555_replicate_analyze.py",
        "--slab-root",
        str(slab),
        "--replicates",
        "42:137,7:11,19:23",
        "--n-boot",
        "25",
        "--out",
        str(out),
    ]
    subprocess.run(cmd, env={**os.environ}, check=True, cwd=REPO_ROOT)
    payload = json.loads(out.read_text())
    assert payload["verdict"]["verdict"] in {"FALSIFIED", "STANDS", "INDETERMINATE"}
    assert payload["verdict"]["n_replicates"] == 3
    for rep_key, rep in payload["per_replicate"].items():
        assert rep["n_rows"] == 432, (rep_key, rep["n_rows"])
        ps = rep["pooled_fit"]["partial_spearman"]
        assert set(ps) == {
            "d_source",
            "d_nearest_neg_nd",
            "shadow_angle",
            "base_prior_marker",
            "training_step",
            "source_delta_g",
        }, set(ps)
        assert rep["family5_holm_primary"], rep_key
        assert rep["z_agreement"]["available"] is True, rep_key
    flags = payload["manifest_flags"]
    assert flags["excluded_bandctrl_dirs"] == ["c504v3_near_seed7_bandctrl"], flags
    assert "c504v3_near_seed7_bandctrl" not in flags["per_cell"], flags
    assert payload["parent_reference"]["available"] is True
    assert (
        abs(payload["parent_reference"]["partial_spearman"]["d_nearest_neg_nd"]["rho"] - 0.110)
        < 0.01
    ), payload["parent_reference"]
    # Per-replicate side files persisted (checkpoint-per-phase).
    side = sorted(slab.glob("analysis_replicate_R*.json"))
    assert len(side) == 3, side
    log.info(
        "PHASE E PASS — analysis: 3 replicates × 432 rows, verdict=%s, "
        "parent step-5 reference machine-read (ρ_nn=%+.3f), _bandctrl excluded",
        payload["verdict"]["verdict"],
        payload["parent_reference"]["partial_spearman"]["d_nearest_neg_nd"]["rho"],
    )
    return out


def phase_f_figures(log, slab: Path, work: Path) -> None:
    out_dir = work / "figures"
    cmd = [
        "uv",
        "run",
        "python",
        "scripts/i555_make_figures.py",
        "--slab-root",
        str(slab),
        "--analysis",
        str(slab / "analysis_replicates.json"),
        "--out-dir",
        str(out_dir),
    ]
    subprocess.run(cmd, env={**os.environ}, check=True, cwd=REPO_ROOT)
    expected = [
        "replicate_forest_nn_shadow",
        "predictor_dot_table",
        "source_dg_strip",
        "raw_scatter_dg_vs_dnn",
        "pooled_vs_replicates",
        "z_agreement",
        "bystander_gate_panel",
    ]
    for stem in expected:
        for ext in ("png", "pdf", "meta.json"):
            p = out_dir / f"{stem}.{ext}"
            assert p.exists() and p.stat().st_size > 0, p
    log.info("PHASE F PASS — figures: %d stems × 3 files at %s", len(expected), out_dir)


def main() -> int:
    log = setup_log()
    keep = os.environ.get("I555_SMOKE_KEEP_WORK")
    work = Path(keep) if keep else Path(tempfile.mkdtemp(prefix="i555_smoke_"))
    work.mkdir(parents=True, exist_ok=True)
    log.info("workdir: %s", work)

    tiny_dir = build_tiny_qwen(work / "tiny_qwen")
    pool = phase_a_pool_build(log, work)
    snap_dir = phase_b_train_with_hard_stop(log, work, tiny_dir, pool)
    phase_c_selection(log, work, tiny_dir, pool, snap_dir)
    phase_d_eval_z_schema_and_dry_run(log, work, tiny_dir, snap_dir)
    slab = build_fixture_slab(log, work)
    phase_e_analysis(log, slab)
    phase_f_figures(log, slab, work)
    log.info("ALL PHASES PASS — i555 local CPU smoke complete (workdir kept: %s)", work)
    return 0


if __name__ == "__main__":
    sys.exit(main())
