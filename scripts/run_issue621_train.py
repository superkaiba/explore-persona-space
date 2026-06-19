"""Issue #621 unified smoke + sweep training dispatcher (rank-1 read/write LoRA).

Forked from ``scripts/run_issue538_train.py`` (pinned ``e6b195f81``;
the #538 pipeline is NOT on main) with the plan §4.2 deltas. Unified per
SKILL.md Step 6d.0: smoke IS sweep with ``--phase smoke`` (the single §7
smoke cell: read arm, florist, seed 42) — same code path, same cell loop,
same env injection, same WandB logging surface, same auto-upload to HF,
same band-stop. PASS_UNIFIED: every downstream phase (eval, analysis,
sentinel) enumerates cells from the SAME ``<out-root>/sweep|anchor_smoke``
cell JSONs this dispatcher writes, so a smoke subset shapes every phase.

Per (placement-arm, source, seed) cell:
  1. Loads persona_bank.json + R_persona/ (sha256-pinned inputs; the pins
     are asserted by run_issue621_preflight.py BEFORE this dispatcher runs).
  2. Builds the per-(source, seed) training JSONL via ``build_cell_rows``
     (strict 1:1, UNIFIED_NEGATIVE_PANEL, realized-disjointness HARD
     assert). The mix is arm-independent: read/write/bridge cells of one
     (source, seed) train on byte-identical rows by construction.
  3. Calls ``train_lora()`` with the §4.2 recipe (rank-1 rsLoRA r=1/α=8,
     per-arm ``lora_targets``, lr=5e-6, epochs cap 16, marker-only loss,
     band-stop [5,12] nat, 10-step checkpoints, A-init snapshot via
     ``save_initial_adapter=True``, per-cell band trajectory JSON).
  4. Verifies the A-init snapshot (exists; lora_B exactly zero;
     ‖a_trained − a_init‖ > 0) — plan §7 snapshot sanity, run per cell.
  5. Smoke phase only: bystander argmax-rate + log P(marker) probe on the
     UNIFIED_NEGATIVE_PANEL, then the §7 gate verdict.
  6. Auto-uploads adapter + adapter_init + checkpoint ladder to HF
     (``adapters/issue_621/<cell_slug>``) per upload-policy.

Plan §14 duty 1: the #538 byte-identity training-mix preflight gate is NOT
inherited (this design's mixes legitimately differ — unified panel,
singleton sources); the R_persona + question-pool SHA-256 prefetch pins ARE
kept (run_issue621_preflight.py + question_pool.py). The pinned data
revision ``HF_TRAIN_MIX_READ_REVISION`` is recorded in every cell result.

CLI (smoke ≡ sweep with one cell):
    uv run python scripts/run_issue621_train.py --phase smoke
    uv run python scripts/run_issue621_train.py --phase sweep --shard 0 --num-shards 4 --gpu-id 0
    uv run python scripts/run_issue621_train.py --phase smoke --dispatcher-dry-run \\
        --allow-smoke-fallback
"""

# ruff: noqa: RUF002  # math/scientific notation in docstrings

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_621 import (
    BASE_MODEL,
    HF_ADAPTER_PATH_PREFIX,
    HF_MODEL_REPO,
    HF_TRAIN_MIX_READ_REVISION,
    IM_END_ID,
    MARKER_ID,
    MARKER_TEXT,
    PLACEMENT_ARMS,
    RECIPE_BAND_HIGH_NATS,
    RECIPE_BAND_LOW_NATS,
    RECIPE_EPOCHS_CAP,
    RECIPE_GRAD_ACCUM,
    RECIPE_LORA_ALPHA,
    RECIPE_LORA_DROPOUT,
    RECIPE_LORA_R,
    RECIPE_LR_PRIMARY,
    RECIPE_MAX_LENGTH,
    RECIPE_PER_DEVICE_BATCH,
    RECIPE_SAVE_STEPS,
    RECIPE_WARMUP_RATIO,
    SMOKE_CELL,
    SOURCES,
    UNIFIED_NEGATIVE_PANEL,
    cell_slug,
    enumerate_cells,
    parse_cell_slug,
)
from explore_persona_space.experiments.issue_621.data_build import (
    build_cell_rows,
    write_rows_jsonl,
)
from explore_persona_space.experiments.issue_621.persona_registry import (
    assert_registry_resolves,
    load_persona_bank,
)
from explore_persona_space.experiments.issue_621.question_pool import load_question_pool

log = logging.getLogger("issue_621.train")

# Parent recipe pin (plan §2/§10): the #538 pipeline this driver forks.
PARENT_PIN_SHA = "e6b195f81"


def _make_train_state_recorder():
    """Capture the trainer's realized + planned step counts at train end.

    Deferred TrainerCallback subclass (keeps the script importable without
    transformers). Round-5 fix: this recorder NO LONGER subscribes to
    ``on_log`` for the band metrics — ``MarkerBandStopCallback`` emits ALL
    its metrics via direct ``wandb.log(...)``, which never flows through
    the trainer's log pipeline, so an ``on_log`` subscription reads nothing
    and unconditionally classified a FALSE band miss (round-4 incident:
    cap-16 AND cap-32 smokes both band-stopped IN BAND per WandB — 5.08 /
    5.36 nat — while the recorder wrote ``fired: false``). The band
    outcome is now derived post-train from the callback's AUTHORITATIVE
    artifact ``band_trajectory.json`` (``derive_band_stop_result`` below);
    this recorder only captures ``state.global_step`` (realized) and
    ``state.max_steps`` (planned) — the early-stop test inputs, also read
    by the §14 duty-2 re-projection via ``global_step_end``.
    """
    from transformers import TrainerCallback

    class _Recorder(TrainerCallback):
        def __init__(self):
            self.global_step_end: int | None = None
            self.max_steps: int | None = None

        def on_train_end(self, args, state, control, **kwargs):
            """Capture the realized global step + the trainer's planned max_steps."""
            self.global_step_end = int(state.global_step)
            # state.max_steps is set by the Trainer at train start (epochs
            # cap x steps/epoch under no explicit max_steps) — the robust
            # planned-schedule source for the early-stop test.
            self.max_steps = int(state.max_steps)

    return _Recorder()


def derive_band_stop_result(
    *,
    trajectory: dict,
    global_step_end: int,
    planned_max_steps: int,
    low_nats: float,
    high_nats: float,
) -> dict:
    """Derive the band-stop outcome from ``band_trajectory.json`` content.

    Pure function (CPU-smoke-testable on synthetic trajectories — see
    ``i621_cpu_smoke.py::smoke_band_stop_derivation``). The trajectory is
    the callback's authoritative artifact (schema
    ``marker_band_trajectory_v1``, atomically rewritten at every probe).

    Derivation:
      - ``final_delta_nats`` = last ``delta_nats`` entry (assert non-empty).
      - ``fired`` = training stopped early (``global_step_end <
        planned_max_steps``) AND ``final_delta_nats >= low_nats`` — the
        callback's stop event reconstructed from its observable effects
        (the callback's stop predicate only fires in-band, and a fired stop
        terminates training right after the firing probe).
      - ``step`` = last trajectory step when fired, else None.

    The in-band classification (``low <= final <= high``) stays in
    ``_smoke_summarize``: an early stop whose final delta overshot past
    ``high_nats`` keeps ``fired=True`` here but classifies as not-in-band
    there.
    """
    if trajectory.get("schema") != "marker_band_trajectory_v1":
        raise AssertionError(
            f"band trajectory schema mismatch: {trajectory.get('schema')!r} "
            "(expected marker_band_trajectory_v1)"
        )
    steps = trajectory.get("steps") or []
    deltas = trajectory.get("delta_nats") or []
    if not deltas or len(deltas) != len(steps):
        raise AssertionError(
            f"band trajectory has no usable probe records (len(steps)={len(steps)}, "
            f"len(delta_nats)={len(deltas)}) — MarkerBandStopCallback appends one "
            "record per probe; an empty trajectory means the callback never probed."
        )
    if planned_max_steps <= 0:
        raise AssertionError(f"planned_max_steps must be > 0, got {planned_max_steps}")
    if global_step_end <= 0:
        raise AssertionError(f"global_step_end must be > 0, got {global_step_end}")
    final_delta = float(deltas[-1])
    stopped_early = bool(global_step_end < planned_max_steps)
    fired = bool(stopped_early and final_delta >= float(low_nats))
    return {
        "fired": fired,
        "step": int(steps[-1]) if fired else None,
        "final_delta_nats": final_delta,
        "band_low_nats": float(low_nats),
        "band_high_nats": float(high_nats),
        "global_step_end": int(global_step_end),
        "planned_max_steps": int(planned_max_steps),
        "stopped_early": stopped_early,
        "source": "band_trajectory.json",
    }


def _derive_and_write_band_stop_result(
    *,
    cell_dir: Path,
    global_step_end: int | None,
    planned_max_steps: int | None,
    low_nats: float,
    high_nats: float,
) -> dict:
    """Read ``band_trajectory.json`` under ``cell_dir``, derive, persist.

    Writes ``marker_band_stop_result.json`` with the SAME schema keys the
    downstream consumers already read (pipeline branch + smoke-gate
    re-projection ``global_step_end`` + uploader glob): ``fired`` /
    ``step`` / ``final_delta_nats`` / ``band_low_nats`` /
    ``band_high_nats`` / ``global_step_end``, plus ``source`` /
    ``planned_max_steps`` / ``stopped_early`` provenance. Only the
    DERIVATION changed in round 5 (trajectory artifact, not on_log).
    """
    traj_path = cell_dir / "band_trajectory.json"
    if not traj_path.is_file():
        raise AssertionError(
            f"band_trajectory.json missing at {traj_path} — MarkerBandStopCallback "
            "writes it atomically at every probe; cannot derive the band verdict."
        )
    if global_step_end is None or planned_max_steps is None:
        raise AssertionError(
            "train-state recorder never saw on_train_end "
            f"(global_step_end={global_step_end}, planned_max_steps={planned_max_steps})"
        )
    payload = derive_band_stop_result(
        trajectory=json.loads(traj_path.read_text()),
        global_step_end=global_step_end,
        planned_max_steps=planned_max_steps,
        low_nats=low_nats,
        high_nats=high_nats,
    )
    out = cell_dir / "marker_band_stop_result.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    log.info(
        "Band-stop result derived from %s: fired=%s delta=%s steps=%d/%d -> %s",
        traj_path.name,
        payload["fired"],
        payload["final_delta_nats"],
        payload["global_step_end"],
        payload["planned_max_steps"],
        out,
    )
    return payload


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_r_persona(out_dir: Path) -> dict[str, dict[str, str]]:
    """Load every persona's R_persona JSON under ``out_dir``.

    Returns ``{persona: {question: response}}``. Content identity vs the
    planning-time verified copies is asserted upstream by
    ``run_issue621_preflight.py`` (sha256 pins).
    """
    if not out_dir.is_dir():
        raise FileNotFoundError(
            f"R_persona dir missing at {out_dir}; run "
            "scripts/run_issue621_preflight.py first (it downloads + sha-pins "
            "the #527 R_persona files)."
        )
    out: dict[str, dict[str, str]] = {}
    for json_path in sorted(out_dir.glob("*.json")):
        payload = json.loads(json_path.read_text())
        # READ schema — R_persona JSONs are inherited byte-identical from #527.
        if payload.get("schema_version") != "issue_527_R_persona_v1":
            raise AssertionError(f"{json_path} R_persona schema mismatch")
        out[payload["persona"]] = payload["responses"]
    return out


def _verify_a_init_snapshot(cell_dir: Path) -> dict:
    """Plan §7 snapshot sanity: adapter_init exists, B_init zero, ‖Δa‖ > 0.

    Reads both safetensors files on CPU. Returns a small dict persisted in
    the cell result. Fails LOUD on any violation — a broken A-init makes
    the H2 ungated-write control unrunnable.
    """
    from safetensors.torch import load_file

    init_path = cell_dir / "adapter_init" / "adapter_model.safetensors"
    final_path = cell_dir / "adapter_model.safetensors"
    if not init_path.is_file():
        raise AssertionError(f"A-init snapshot missing at {init_path}")
    if not final_path.is_file():
        raise AssertionError(f"final adapter missing at {final_path}")
    init_sd = load_file(str(init_path))
    final_sd = load_file(str(final_path))
    if set(init_sd.keys()) != set(final_sd.keys()):
        raise AssertionError(
            f"adapter_init keys != final adapter keys "
            f"(init-only: {sorted(set(init_sd) - set(final_sd))[:3]}, "
            f"final-only: {sorted(set(final_sd) - set(init_sd))[:3]})"
        )
    b_max_abs = 0.0
    delta_a_norm_total = 0.0
    a_init_norm_total = 0.0
    n_a = n_b = 0
    for key, init_t in init_sd.items():
        if "lora_B" in key:
            b_max_abs = max(b_max_abs, float(init_t.abs().max().item()))
            n_b += 1
        elif "lora_A" in key:
            delta = (final_sd[key].float() - init_t.float()).norm().item()
            delta_a_norm_total += float(delta)
            a_init_norm_total += float(init_t.float().norm().item())
            n_a += 1
    if n_a == 0 or n_b == 0:
        raise AssertionError(f"adapter_init has no lora_A/lora_B tensors (n_a={n_a}, n_b={n_b})")
    if b_max_abs != 0.0:
        raise AssertionError(
            f"B_init is NOT exactly zero (max |B_init| = {b_max_abs:g}) — PEFT "
            "default init contract violated; the A-init control is invalid."
        )
    if delta_a_norm_total <= 0.0:
        raise AssertionError(
            "norm(a_trained - a_init) == 0 across all modules - either training "
            "did not update A at all (snapshot taken after training?) or the "
            "final adapter equals the init. Snapshot sanity FAILED."
        )
    return {
        "n_lora_A_tensors": n_a,
        "n_lora_B_tensors": n_b,
        "b_init_max_abs": b_max_abs,
        "delta_a_norm_total": delta_a_norm_total,
        "a_init_norm_total": a_init_norm_total,
    }


def _build_smoke_probe_rows(
    *,
    panel: tuple[str, ...],
    persona_bank: dict[str, str],
    questions: list[str],
    r_persona: dict[str, dict[str, str]],
    tokenizer,
    n_probe_per_persona: int = 8,
) -> dict[str, list[tuple[str, list[int], int]]]:
    """Pre-build per-bystander probe inputs (full_ids + post-response slot).

    For each persona in the UNIFIED_NEGATIVE_PANEL, sample
    ``n_probe_per_persona`` questions and tokenize
    ``T_persona(q) + R_persona(q)``. Returns
    ``{persona: [(question, full_ids, post_response_slot), ...]}``.
    """
    from explore_persona_space.experiments.issue_621.shift_extract import (
        _resolve_post_response_slot,
    )

    out: dict[str, list[tuple[str, list[int], int]]] = {}
    rng = np.random.default_rng(0)
    for persona in panel:
        if persona not in r_persona:
            raise AssertionError(
                f"R_persona missing for negative persona {persona!r}; preflight gap."
            )
        n = min(n_probe_per_persona, len(questions))
        idxs = rng.choice(len(questions), size=n, replace=False)
        rows: list[tuple[str, list[int], int]] = []
        for i in idxs:
            q = questions[int(i)]
            if q not in r_persona[persona]:
                raise AssertionError(f"R_persona[{persona!r}] missing q={q!r}")
            messages = [
                {"role": "system", "content": persona_bank[persona]},
                {"role": "user", "content": q},
                {"role": "assistant", "content": r_persona[persona][q]},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            full_ids = tokenizer.encode(text, add_special_tokens=False)
            slot = _resolve_post_response_slot(tokenizer, messages[:2], full_ids)
            rows.append((q, full_ids, slot))
        out[persona] = rows
    return out


def _bystander_headroom_probe(
    *,
    base_model_path: str,
    adapter_dir: str,
    probe_rows: dict[str, list[tuple[str, list[int], int]]],
    device: str = "cuda:0",
) -> dict[str, dict[str, float]]:
    """Forward-only probe: per bystander, argmax-rate at slot + mean Δ log P(marker).

    Returns ``{bystander: {"argmax_rate": float, "delta_logp_mean": float,
    "logp_trained_mean": float, "logp_base_mean": float}}``.
    """
    import torch
    import torch.nn.functional as F
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    log.info("Loading base model (%s) for bystander-headroom probe", base_model_path)
    base = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    ).eval()

    log.info("Loading trained adapter from %s", adapter_dir)
    trained = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    trained = PeftModel.from_pretrained(trained, adapter_dir).eval()

    out: dict[str, dict[str, float]] = {}
    for persona, rows in probe_rows.items():
        argmax_hits = 0
        delta_acc = 0.0
        logp_trained_acc = 0.0
        logp_base_acc = 0.0
        for _q, full_ids, slot in rows:
            ids = torch.tensor([full_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                out_base = base(ids)
                out_tr = trained(ids)
            lp_base = F.log_softmax(out_base.logits[0, slot - 1].float(), dim=-1)
            lp_tr = F.log_softmax(out_tr.logits[0, slot - 1].float(), dim=-1)
            if int(out_tr.logits[0, slot - 1].argmax().item()) == MARKER_ID:
                argmax_hits += 1
            delta_acc += float((lp_tr[MARKER_ID] - lp_base[MARKER_ID]).item())
            logp_trained_acc += float(lp_tr[MARKER_ID].item())
            logp_base_acc += float(lp_base[MARKER_ID].item())
        n = len(rows)
        out[persona] = {
            "argmax_rate": argmax_hits / n,
            "delta_logp_mean": delta_acc / n,
            "logp_trained_mean": logp_trained_acc / n,
            "logp_base_mean": logp_base_acc / n,
        }

    # Free GPU; the next cell needs a fresh allocation.
    del base
    del trained
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return out


def _train_one_cell(
    *,
    arm: str,
    source: str,
    seed: int,
    persona_bank: dict[str, str],
    questions: list[str],
    r_persona: dict[str, dict[str, str]],
    lr: float,
    epochs_cap: int,
    gpu_id: int,
    output_root: Path,
    hf_path_in_repo: str,
    band_stop_low: float,
    band_stop_high: float,
    dispatcher_dry_run: bool,
) -> tuple[str, float, dict]:
    """Train one (arm, source, seed) cell.

    Returns (output_dir, final_train_loss, extras) where extras carries the
    realized wall seconds + A-init verification payload.
    """
    from transformers import AutoTokenizer

    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    slug = cell_slug(arm, source, seed)
    cell_dir = output_root / "cells" / slug
    cell_dir.mkdir(parents=True, exist_ok=True)

    data_dir = output_root / "training_mixes"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Arm-independent mix: one JSONL per (source, seed), shared by all arms.
    train_path = data_dir / f"{source}__seed{seed}.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Marker token preflight (in-process, per marker-leakage rule; thread
    # with shlex.quote for any shell layer).
    marker_quoted = shlex.quote(MARKER_TEXT)
    log.info("Marker token (shlex-quoted) = %s", marker_quoted)
    encoded = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if encoded != [MARKER_ID]:
        raise AssertionError(
            f"Marker token drift inside _train_one_cell: {encoded} != [{MARKER_ID}]"
        )
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end != IM_END_ID:
        raise AssertionError(f"<|im_end|> drift: {im_end} != {IM_END_ID}")

    if train_path.is_file():
        log.info("Reusing existing mix %s (arm-independent, deterministic per seed)", train_path)
    else:
        log.info("Building training rows: source=%s seed=%d (arm-independent)", source, seed)
        rows = build_cell_rows(
            source=source,
            persona_bank=persona_bank,
            questions=questions,
            r_persona=r_persona,
            tokenizer=tokenizer,
            seed=seed,
        )
        log.info("Writing %d training rows to %s", len(rows), train_path)
        write_rows_jsonl(rows, train_path)

    if dispatcher_dry_run:
        log.warning(
            "--dispatcher-dry-run: SKIPPING train_lora() call. "
            "All pre-GPU plumbing exercised: tokenizer load + marker assert + "
            "rows build (incl. realized-disjointness assert) + JSONL write + "
            "cell dir create."
        )
        return str(cell_dir), 0.0, {"wall_s": 0.0, "a_init_check": None}

    cfg = TrainLoraConfig(
        gpu_id=gpu_id,
        epochs=epochs_cap,
        lr=lr,
        lora_r=RECIPE_LORA_R,
        lora_alpha=RECIPE_LORA_ALPHA,
        lora_dropout=RECIPE_LORA_DROPOUT,
        lora_targets=list(PLACEMENT_ARMS[arm]),
        batch_size=RECIPE_PER_DEVICE_BATCH,
        grad_accum=RECIPE_GRAD_ACCUM,
        max_length=RECIPE_MAX_LENGTH,
        warmup_ratio=RECIPE_WARMUP_RATIO,
        seed=seed,
        run_name=f"issue621_{slug}",
        report_to="wandb",
        save_strategy="steps",
        save_steps=RECIPE_SAVE_STEPS,
        save_only_model=True,
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        marker_suppress_at_post_response_slot=True,
        marker_im_end_token_id=IM_END_ID,
        marker_band_stop=True,
        marker_band_low_nats=band_stop_low,
        marker_band_high_nats=band_stop_high,
        marker_band_trajectory_path=str(cell_dir / "band_trajectory.json"),
        save_initial_adapter=True,
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=hf_path_in_repo,
    )

    # MooseFS quota safety per CLAUDE.md gotchas.
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    # Delete-after-eval adapter-persist recipe (upload-policy.md).
    os.environ["EPM_PERSIST_ADAPTER_HF_REPO"] = HF_MODEL_REPO
    os.environ["EPM_PERSIST_ADAPTER_SUBFOLDER"] = hf_path_in_repo

    recorder = _make_train_state_recorder()

    t0 = time.monotonic()
    output_dir, train_loss = train_lora(
        BASE_MODEL, str(train_path), str(cell_dir), cfg=cfg, callbacks=[recorder]
    )
    wall_s = time.monotonic() - t0

    # Round-5 fix: derive the band outcome from the callback's authoritative
    # band_trajectory.json (NOT from on_log keys the callback never routes
    # through the trainer log pipeline) and persist it for the downstream
    # consumers (pipeline branch, smoke gate, §14 duty-2 re-projection).
    _derive_and_write_band_stop_result(
        cell_dir=Path(cell_dir),
        global_step_end=recorder.global_step_end,
        planned_max_steps=recorder.max_steps,
        low_nats=band_stop_low,
        high_nats=band_stop_high,
    )

    # Plan §7 snapshot sanity — per cell (cheap CPU read of two safetensors).
    a_init_check = _verify_a_init_snapshot(Path(output_dir))

    log.info(
        "TRAIN DONE cell=%s loss=%.4f wall=%.0fs -> %s (uploaded to %s)",
        slug,
        train_loss,
        wall_s,
        output_dir,
        f"{HF_MODEL_REPO}/{hf_path_in_repo}",
    )
    return output_dir, train_loss, {"wall_s": wall_s, "a_init_check": a_init_check}


def _smoke_summarize(
    *,
    smoke_results: list[dict],
    band_low: float,
    band_high: float,
    bystander_argmax_max: float = 0.92,
) -> dict:
    """Apply the §7 smoke-gate verdict for the SINGLE smoke cell.

    PASS requires ALL of:
      (i)   band-stop fired with final source ΔG ∈ [band_low, band_high]
            (both derived from the callback's band_trajectory.json via
            derive_band_stop_result — round-5 fix; an early stop whose
            final ΔG overshot past band_high keeps fired=True but
            classifies here as not-in-band);
      (ii)  ALL 4 negative-panel personas below the argmax ceiling (< 0.92);
      (iii) A-init snapshot sanity (adapter_init exists, B_init exactly
            zero, ‖Δa‖ > 0) — enforced fail-loud inside _train_one_cell,
            recorded here;
      (iv)  band trajectory file exists with ≥1 logged probe point (guard
            smoke-verifiability, plan §4.2).

    The §7 off-line-eval parity assert (in-loop ΔG vs eval ΔG within 1 nat,
    #534) runs in the PIPELINE after the smoke eval subprocess — see
    scripts/i621_smoke_gate.py.

    A band miss (gate i) is the cheap intermediate signal for the §7
    fallback path (raise the cap once to 32 and re-smoke) — surfaced in the
    verdict as ``band_missed`` so the pipeline can act on it.
    """
    assert len(smoke_results) == 1, f"smoke phase expects 1 cell, got {len(smoke_results)}"
    r = smoke_results[0]
    final_delta = r.get("final_source_delta_nats")
    band_ok = (
        r.get("band_stop_fired") is True
        and final_delta is not None
        and band_low <= float(final_delta) <= band_high
    )
    bys = r.get("bystander_probe", {})
    bys_ok = bool(bys) and all(m["argmax_rate"] < bystander_argmax_max for m in bys.values())
    a_init_ok = bool(r.get("a_init_check"))

    traj_path = Path(r["output_dir"]) / "band_trajectory.json"
    traj_ok = False
    traj_points = 0
    if traj_path.is_file():
        try:
            traj = json.loads(traj_path.read_text())
            records = traj.get("records", traj if isinstance(traj, list) else [])
            traj_points = len(records)
            traj_ok = traj_points >= 1
        except (json.JSONDecodeError, OSError):
            traj_ok = False

    verdict = "PASS" if (band_ok and bys_ok and a_init_ok and traj_ok) else "FAIL"
    return {
        "cell_slug": r["cell_slug"],
        "verdict": verdict,
        "band_ok": band_ok,
        "band_missed": not band_ok,
        "band_stop_fired": r.get("band_stop_fired"),
        "bystanders_ok": bys_ok,
        "a_init_ok": a_init_ok,
        "trajectory_ok": traj_ok,
        "trajectory_points": traj_points,
        "final_source_delta_nats": r.get("final_source_delta_nats"),
        "band_stop_step": r.get("band_stop_step"),
        "global_step_end": r.get("global_step_end"),
        "train_wall_s": r.get("train_wall_s"),
        "band_low_nats": band_low,
        "band_high_nats": band_high,
        "bystander_argmax_max": bystander_argmax_max,
    }


def _resolve_cells(args) -> list[tuple[str, str, int]]:
    """Resolve the (arm, source, seed) cells for this invocation.

    Smoke = the single SMOKE_CELL (overridable via --cells for canary
    needs). Sweep = the full 30-cell grid minus SMOKE_CELL (already trained
    by the smoke phase), filtered by --arm/--source/--seed/--cells, then
    deterministically sharded by --shard/--num-shards.
    """
    if args.cells:
        explicit = [parse_cell_slug(s) for s in args.cells]
        return explicit

    if args.phase == "smoke":
        return [SMOKE_CELL]

    cells = [c for c in enumerate_cells() if c != SMOKE_CELL or args.include_smoke_cell]
    if args.arm:
        cells = [c for c in cells if c[0] == args.arm]
    if args.source:
        cells = [c for c in cells if c[1] == args.source]
    if args.seed is not None:
        cells = [c for c in cells if c[2] == args.seed]
    if args.num_shards > 1:
        cells = [c for i, c in enumerate(cells) if i % args.num_shards == args.shard]
    return cells


def main(argv: list[str] | None = None) -> int:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", required=True, choices=["smoke", "sweep"])
    ap.add_argument(
        "--r-persona-dir",
        # INHERITED READ from #527 (sha256-pinned by run_issue621_preflight.py).
        default="eval_results/issue_527/R_persona",
    )
    ap.add_argument(
        "--out-root",
        # NEW WRITE namespace.
        default="eval_results/issue_621",
    )
    ap.add_argument("--arm", choices=sorted(PLACEMENT_ARMS), default=None)
    ap.add_argument("--source", choices=SOURCES, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--cells",
        nargs="+",
        default=None,
        help="Explicit cell slugs (e.g. r1_read__florist__seed42). Overrides filters.",
    )
    ap.add_argument(
        "--include-smoke-cell",
        action="store_true",
        help="Sweep only — also (re)train the smoke cell instead of skipping it.",
    )
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument(
        "--lr",
        type=float,
        default=RECIPE_LR_PRIMARY,
        help=(
            f"Default {RECIPE_LR_PRIMARY}. NO autonomous lr retry — the marker "
            "recipe forbids lr>5e-6 (strength via steps, never lr)."
        ),
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=RECIPE_EPOCHS_CAP,
        help="Epochs cap (16; ONE authorized raise to 32 on a smoke band miss, plan §7/§13).",
    )
    ap.add_argument("--band-low-nats", type=float, default=RECIPE_BAND_LOW_NATS)
    ap.add_argument("--band-high-nats", type=float, default=RECIPE_BAND_HIGH_NATS)
    ap.add_argument("--n-questions", type=int, default=400)
    ap.add_argument(
        "--allow-smoke-fallback",
        action="store_true",
        help="Permit the 20-question smoke fallback (smoke only).",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a cell whose result JSON already exists (idempotent re-runs).",
    )
    ap.add_argument(
        "--dispatcher-dry-run",
        action="store_true",
        help=(
            "Stub out train_lora() and the bystander probe — exercises the "
            "pre-GPU plumbing (data load, marker assert, rows build incl. "
            "disjointness assert, JSONL write, cell-dir create) without CUDA."
        ),
    )
    args = ap.parse_args(argv)

    if not 0 <= args.shard < args.num_shards:
        raise SystemExit(f"--shard {args.shard} out of range for --num-shards {args.num_shards}")

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    log.info("Loading persona-bank + R_persona (pinned inputs)")
    persona_bank = load_persona_bank()
    assert_registry_resolves(persona_bank)
    r_persona = _load_r_persona(Path(args.r_persona_dir))
    log.info("Loaded %d personas of R_persona", len(r_persona))

    questions = load_question_pool(
        n_required=args.n_questions, allow_smoke_fallback=args.allow_smoke_fallback
    )

    cells = _resolve_cells(args)
    log.info(
        "Phase %s: %d cell(s) on shard %d/%d: %s",
        args.phase,
        len(cells),
        args.shard,
        args.num_shards,
        [cell_slug(*c) for c in cells],
    )

    git_commit = _git_commit()
    timestamp = _dt.datetime.now(tz=_dt.UTC).isoformat(timespec="seconds")
    # Plan §14 duty 1: surface the pinned data revision in run metadata.
    log.info("HF_TRAIN_MIX_READ_REVISION=%s (pinned input revision)", HF_TRAIN_MIX_READ_REVISION)

    smoke_tokenizer = None
    smoke_probe_rows = None
    if args.phase == "smoke" and not args.dispatcher_dry_run:
        from transformers import AutoTokenizer

        smoke_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        smoke_probe_rows = _build_smoke_probe_rows(
            panel=UNIFIED_NEGATIVE_PANEL,
            persona_bank=persona_bank,
            questions=questions,
            r_persona=r_persona,
            tokenizer=smoke_tokenizer,
            n_probe_per_persona=8,
        )

    phase_dir = out_root / ("anchor_smoke" if args.phase == "smoke" else "sweep")
    phase_dir.mkdir(parents=True, exist_ok=True)

    smoke_results: list[dict] = []
    for arm, source, seed in cells:
        slug = cell_slug(arm, source, seed)
        cell_path = phase_dir / f"{slug}.json"
        if args.skip_existing and cell_path.is_file():
            log.info("Skipping %s (result exists at %s)", slug, cell_path)
            continue
        hf_subfolder = f"{HF_ADAPTER_PATH_PREFIX}/{slug}"
        log.info(
            "[phase=train_cell] cell=%s targets=%s lr=%g epochs_cap=%d",
            slug,
            PLACEMENT_ARMS[arm],
            args.lr,
            args.epochs,
        )

        try:
            out_dir, loss, extras = _train_one_cell(
                arm=arm,
                source=source,
                seed=seed,
                persona_bank=persona_bank,
                questions=questions,
                r_persona=r_persona,
                lr=args.lr,
                epochs_cap=args.epochs,
                gpu_id=args.gpu_id,
                output_root=out_root,
                hf_path_in_repo=hf_subfolder,
                band_stop_low=args.band_low_nats,
                band_stop_high=args.band_high_nats,
                dispatcher_dry_run=args.dispatcher_dry_run,
            )
        except Exception as e:
            # Per CLAUDE.md fail-fast: log + re-raise so the launcher's
            # set -e aborts the pipeline. No silent skip.
            log.exception("cell=%s training crashed: %s", slug, e)
            raise

        cell_result: dict = {
            "cell_slug": slug,
            "arm": arm,
            "source": source,
            "seed": seed,
            "lora_targets": list(PLACEMENT_ARMS[arm]),
            "lora_r": RECIPE_LORA_R,
            "lora_alpha": RECIPE_LORA_ALPHA,
            "lr": args.lr,
            "epochs_cap": args.epochs,
            "band_low_nats": args.band_low_nats,
            "band_high_nats": args.band_high_nats,
            "output_dir": out_dir,
            "hf_subfolder": hf_subfolder,
            "final_train_loss": loss,
            "train_wall_s": extras["wall_s"],
            "a_init_check": extras["a_init_check"],
            "git_commit": git_commit,
            "parent_pin_sha": PARENT_PIN_SHA,
            "hf_train_mix_read_revision": HF_TRAIN_MIX_READ_REVISION,
            "timestamp_utc": timestamp,
            "base_model": BASE_MODEL,
            "negative_panel": list(UNIFIED_NEGATIVE_PANEL),
            "wandb_run_name": f"issue621_{slug}",
        }

        cb_result_path = Path(out_dir) / "marker_band_stop_result.json"
        if cb_result_path.is_file():
            cb = json.loads(cb_result_path.read_text())
            cell_result["band_stop_fired"] = cb.get("fired", False)
            cell_result["final_source_delta_nats"] = cb.get("final_delta_nats")
            cell_result["band_stop_step"] = cb.get("step")
            cell_result["global_step_end"] = cb.get("global_step_end")
        else:
            cell_result["band_stop_fired"] = None
            cell_result["final_source_delta_nats"] = None
            cell_result["band_stop_step"] = None
            cell_result["global_step_end"] = None

        if args.phase == "smoke" and smoke_probe_rows is not None:
            log.info("[phase=smoke_probe] cell=%s — bystander headroom probe", slug)
            bys = _bystander_headroom_probe(
                base_model_path=BASE_MODEL,
                adapter_dir=out_dir,
                probe_rows=smoke_probe_rows,
            )
            cell_result["bystander_probe"] = bys

        # Per-cell checkpoint — write IMMEDIATELY (checkpoint-per-phase).
        cell_path.write_text(json.dumps(cell_result, indent=2))
        log.info("Wrote %s", cell_path)
        smoke_results.append(cell_result)

    if args.phase == "smoke" and smoke_results:
        summary = _smoke_summarize(
            smoke_results=smoke_results,
            band_low=args.band_low_nats,
            band_high=args.band_high_nats,
        )
        summary_path = out_root / "anchor_smoke" / "summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        log.info("Smoke train-side verdict=%s -> %s", summary["verdict"], summary_path)
        if summary["verdict"] != "PASS" and not args.dispatcher_dry_run:
            log.error(
                "Smoke train-side gate FAILED (band_ok=%s bystanders_ok=%s "
                "a_init_ok=%s trajectory_ok=%s) at band [%g, %g] nat. See %s. "
                "On a pure band miss the §7 fallback is ONE cap raise to 32 "
                "(re-run with --epochs 32); NO lr raise (recipe forbids).",
                summary["band_ok"],
                summary["bystanders_ok"],
                summary["a_init_ok"],
                summary["trajectory_ok"],
                args.band_low_nats,
                args.band_high_nats,
                summary_path,
            )
            return 2

    log.info("train dispatcher exit OK (phase=%s, %d cell(s))", args.phase, len(smoke_results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
