#!/usr/bin/env python3
"""Issue #558 persona-panel eval — 5 cells x 4-float marker slot stats per adapter.

Adapted copy of the pinned #543 eval rig (issue-543 branch @
93c410ddcb00ed3417205471821d0c5517a227d3, byte-identical to the parent's
run-time commit 78f0a45d3); plan tasks/.../558/plans/plan.md section 4.2 step 2
enumerates the changes exhaustively. Everything else is verbatim parent
instrument: marker_preflight, gauge asserts (front + worker), ONE FRESH vLLM
engine per adapter, vLLM-free slot-stats subprocess (4 floats x trained AND
base via disable_adapter()), marker-stripping before the slot read,
checkpoint-per-cell persists, summarize_cell three-space rollup,
repro_metadata, [phase=...] log lines + end-of-run sentinel.

Changes vs the pinned eval_issue543.py:
  - build_cells_panel: 5 cells (trigger50 / doctor / software_engineer /
    french_person / police_officer), ALL persona-system-prompt + key-present
    on eval_questions[0:50], greedy, n=50 (plan section 4.1).
  - panel_persona_prompts(): parent's all_persona_prompts() extended with
    police_officer (never trained in the #543 chain).
  - Phase pinned to phase2 (the 12 post-SFT adapters are the objects under
    study); output dir eval_results/issue_558/<arm>/seed<S>/phase2/; HF raw
    bucket issue558_persona_panel/raw_completions/<arm>_seed<S>_phase2;
    LoRA id prefix issue558_.
  - Doctor-cell anchor check vs the parent's committed rollup (adapter-
    application assert, marker-leakage-measurement.md): hard-FAIL only under
    --anchor-gate (the smoke/gate adapter r50_seed42); audit fields + WARN
    otherwise. --skip-anchor for diagnostic re-runs only.
  - --dry-run-cells: CPU-only launch-validity dry run (marker preflight +
    question fetch + cell construction digests + Hub adapter_config gauge
    assert + parent-rollup anchor-reference lookup), exits before any vLLM /
    CUDA import with [phase=done].

Usage (pod, 1 GPU; smoke = the first production cell, full n):
    uv run python scripts/eval_issue558_panel.py --arm r50 --seed 42 --gpu 0 --anchor-gate
    uv run python scripts/eval_issue558_panel.py --arm r25 --seed 137 --gpu 1
VM CPU dry run:
    uv run python scripts/eval_issue558_panel.py --arm r50 --seed 42 --dry-run-cells
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="eval_issue558_panel")

import os  # noqa: E402

from _issue543_common import (  # noqa: E402
    ARMS,
    BASE_MODEL,
    DEFAULT_ASSISTANT_KEY,
    EOS_TOKEN_ID,
    EVAL_MAX_NEW_TOKENS,
    HUB_MODEL_REPO,
    MARKER_TEXT,
    N_SMOKE_PROMPTS,
    PROJECT_ROOT,
    adapter_subfolder,
    all_persona_prompts,
    cell_slug,
    ensure_eval_questions_local,
    marker_preflight,
    phase_log,
    repro_metadata,
    sentinel_dir,
    trigger_user,
    truncated,
)

log = logging.getLogger("eval_issue558_panel")

ISSUE_558 = 558
LOGPROB_BATCH_SIZE = 8

# Panel constants (plan sections 4.1 / 10).
N_PANEL_PROMPTS = 50  # all 5 cells share eval_questions[0:50] (parent doctor-cell shape)
PANEL_PHASE = "phase2"  # the 12 post-SFT adapters are the objects under study
EVAL_RESULTS_DIR_558 = PROJECT_ROOT / "eval_results" / "issue_558"
HUB_RAW_COMPLETIONS_BUCKET_558 = "issue558_persona_panel/raw_completions"
PARENT_ROLLUP_PATH = PROJECT_ROOT / "eval_results" / "issue_543" / "rollup.json"
ANCHOR_TOL_NATS = 1.0  # marker-leakage-measurement.md adapter-application assert (~1 nat)

# Cell slug -> persona key. ALL cells are persona-system-prompt + key-present
# (trigger_user) on the SAME question slice (plan section 4.1).
PANEL_CELLS: dict[str, str] = {
    "trigger50": "assistant",  # within-run no-dip baseline (paired denominator)
    "doctor": "medical_doctor",  # within-run dip reproduction + anchor vs parent
    "software_engineer": "software_engineer",  # trained-negative x non-medical
    "french_person": "french_person",  # trained-negative x non-medical (2nd)
    "police_officer": "police_officer",  # never-trained x non-medical
}


def panel_persona_prompts() -> dict[str, str]:
    """Parent persona map (assistant + 3 trained negatives) + police_officer."""
    from explore_persona_space.personas import PERSONAS

    prompts = all_persona_prompts()
    prompts["police_officer"] = PERSONAS["police_officer"]
    return prompts


# ── Cells (deterministic slice [0:50]; plan section 4.1) ────────────────────


def build_cells_panel(eval_questions: list[str], *, smoke: bool) -> dict[str, list[dict]]:
    """5 panel cells, all on eval_questions[0:n] with the trigger key present.

    n = 50 (N_PANEL_PROMPTS); smoke mode uses 20 (N_SMOKE_PROMPTS, parity with
    the parent). Every cell shares the identical question slice so all paired
    contrasts are same-question by construction.
    """
    personas = panel_persona_prompts()
    n = N_SMOKE_PROMPTS if smoke else N_PANEL_PROMPTS
    qs = eval_questions[:n]
    if len(qs) != n:
        raise RuntimeError(f"Need {n} eval questions; got {len(qs)}")

    cells: dict[str, list[dict]] = {}
    for cell_name, persona_key in PANEL_CELLS.items():
        system = (
            personas[DEFAULT_ASSISTANT_KEY] if persona_key == "assistant" else personas[persona_key]
        )
        cells[cell_name] = [
            {"system": system, "user": trigger_user(q), "persona_key": persona_key, "trigger": True}
            for q in qs
        ]
    empty = [k for k, v in cells.items() if not v]
    if empty:
        raise RuntimeError(f"Empty cell(s) (smoke={smoke}): {empty}")
    return cells


# ── Adapter resolution (verbatim parent; phase pinned to phase2 by caller) ──


def resolve_adapter(arm: str, seed: int, phase: str, adapter_path: str | None) -> Path:
    """Local path if given (dispatcher hand-off); else fetch from HF Hub."""
    if adapter_path:
        p = Path(adapter_path)
        if not p.exists() or not (p / "adapter_config.json").exists():
            raise FileNotFoundError(f"--adapter-path invalid (no adapter_config.json): {p}")
        return p
    from huggingface_hub import snapshot_download

    sub = f"adapters/{adapter_subfolder(arm, seed, phase)}"
    log.info("Resolving adapter from Hub: %s/%s", HUB_MODEL_REPO, sub)
    local = snapshot_download(
        repo_id=HUB_MODEL_REPO,
        allow_patterns=[f"{sub}/*"],
        token=os.environ.get("HF_TOKEN"),
    )
    adapter_dir = Path(local) / sub
    if not adapter_dir.exists() or not (adapter_dir / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter missing/empty on Hub: {adapter_dir}")
    return adapter_dir


def assert_adapter_gauge_free(adapter_dir: Path) -> dict:
    """Run the gauge assert on adapter_config.json BEFORE any logit readout."""
    from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config

    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    assert_gauge_free_adapter_config(cfg, context=str(adapter_dir))
    return cfg


# ── Doctor-cell anchor check vs the parent's committed rollup (plan 4.2) ────


def parent_doctor_reference(arm: str, seed: int) -> dict:
    """Parent's recorded doctor-cell means for this adapter (rollup.json in git)."""
    if not PARENT_ROLLUP_PATH.exists():
        raise FileNotFoundError(
            f"Parent rollup missing: {PARENT_ROLLUP_PATH} (committed in git; sync the repo)"
        )
    parent = json.loads(PARENT_ROLLUP_PATH.read_text())
    cell_key = f"{arm}_seed{seed}"
    try:
        ref = parent["cells"][cell_key]["phases"][PANEL_PHASE]["doctor"]
    except (KeyError, TypeError) as e:
        raise RuntimeError(
            f"Parent rollup has no {PANEL_PHASE} doctor cell for {cell_key}: {e}"
        ) from e
    for k in ("delta_logp_mean", "delta_eos_margin_mean"):
        if k not in ref or not math.isfinite(ref[k]):
            raise RuntimeError(f"Parent doctor reference for {cell_key} missing/non-finite {k}")
    return ref


def check_doctor_anchor(arm: str, seed: int, doctor_summary: dict, *, gate: bool) -> dict:
    """Adapter-application anchor: this run's doctor cell vs the parent's record.

    Tolerance +-1.0 nat on BOTH delta_logp_mean and delta_eos_margin_mean (an
    unapplied adapter reads ~7 nats off — incident #534; version drift reads
    << 1 nat). Hard-raise only when ``gate`` (the smoke/gate adapter
    r50_seed42); otherwise the offsets are audit fields with a WARN on breach
    (plan section 4.2 step 2 scoping — the parent comparison is a declared
    non-load-bearing audit for the other 11 adapters).
    """
    ref = parent_doctor_reference(arm, seed)
    off_logp = doctor_summary["delta_logp_mean"] - ref["delta_logp_mean"]
    off_eosm = doctor_summary["delta_eos_margin_mean"] - ref["delta_eos_margin_mean"]
    breach = abs(off_logp) > ANCHOR_TOL_NATS or abs(off_eosm) > ANCHOR_TOL_NATS
    result = {
        "anchor_checked": True,
        "anchor_gate": gate,
        "anchor_tol_nats": ANCHOR_TOL_NATS,
        "anchor_parent_delta_logp_mean": ref["delta_logp_mean"],
        "anchor_parent_delta_eos_margin_mean": ref["delta_eos_margin_mean"],
        "anchor_offset_logp": off_logp,
        "anchor_offset_eosm": off_eosm,
        "anchor_breach": breach,
    }
    msg = (
        f"Doctor anchor {arm}_seed{seed}: offset_logp={off_logp:+.3f} "
        f"offset_eosm={off_eosm:+.3f} (tol +-{ANCHOR_TOL_NATS}) breach={breach}"
    )
    if breach and gate:
        raise RuntimeError(
            f"ANCHOR GATE FAIL — {msg}. Adapter-application assert breached on the "
            "gate adapter; do NOT launch the sweep. Diagnose per plan section 7 kill "
            "criteria (incl. the PEFT cross-check of adapter application, #492) "
            "before any further GPU spend."
        )
    if breach:
        log.warning("%s — recorded as audit field; run continues (non-gate adapter).", msg)
    else:
        log.info("%s", msg)
    return result


# ── vLLM generation (FRESH engine per adapter; verbatim parent) ─────────────


def _teardown_vllm(llm: Any) -> None:
    """Reap vLLM worker subprocesses (gotchas.md vLLM teardown)."""
    import contextlib
    import gc

    import psutil
    import torch

    try:
        from vllm.distributed.parallel_state import (
            destroy_distributed_environment,
            destroy_model_parallel,
        )

        destroy_model_parallel()
        destroy_distributed_environment()
    except Exception as e:
        log.warning("vllm distributed teardown raised: %s", e)
    with contextlib.suppress(Exception):
        del llm
    gc.collect()
    torch.cuda.empty_cache()
    me = psutil.Process()
    for child in me.children(recursive=True):
        try:
            child.terminate()
            child.wait(timeout=5)
        except Exception:
            with contextlib.suppress(Exception):
                child.kill()


def generate_completions(
    *,
    adapter_dir: Path,
    lora_name: str,
    cells: dict[str, list[dict]],
    out_dir: Path,
) -> dict[str, list[dict]]:
    """Greedy vLLM generation per cell on ONE fresh engine for ONE adapter.

    Adapter isolation (parent rig invariant): the engine is created fresh in
    this invocation, serves exactly one LoRA, and every record logs the
    adapter path + LoRA id.
    """
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    log.info("Loading FRESH vLLM engine: base=%s adapter=%s", BASE_MODEL, adapter_dir)
    llm = LLM(
        model=BASE_MODEL,
        tensor_parallel_size=1,
        dtype="bfloat16",
        max_model_len=4096,
        max_num_seqs=64,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=16,
        # 0.70 inherited from the parent rig (comment there documents the
        # 2026-06-10 smoke-cell incident); conservative-but-ample on a
        # dedicated eval GPU (model ~15 GiB + ~40 GiB KV at max_num_seqs=64).
        gpu_memory_utilization=0.70,
    )
    lora_req = LoRARequest(lora_name, 1, str(adapter_dir))
    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=EVAL_MAX_NEW_TOKENS, n=1)

    out: dict[str, list[dict]] = {}
    try:
        for cell_name, items in cells.items():
            prefixes = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": it["system"]},
                        {"role": "user", "content": it["user"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for it in items
            ]
            log.info("Generating cell=%s n=%d", cell_name, len(prefixes))
            responses = llm.generate(prefixes, sampling, lora_request=lora_req)
            recs: list[dict] = []
            for it, prefix, resp in zip(items, prefixes, responses, strict=True):
                g = resp.outputs[0]
                recs.append(
                    {
                        **it,
                        "prefix": prefix,
                        "completion_text": g.text,
                        "n_generated_tokens": len(g.token_ids),
                        "truncated": truncated(len(g.token_ids), EVAL_MAX_NEW_TOKENS),
                        "contains_marker": MARKER_TEXT in g.text,
                        "ends_with_marker": g.text.rstrip().endswith(MARKER_TEXT.strip()),
                        "adapter_path": str(adapter_dir),
                        "lora_id": lora_name,
                    }
                )
            out[cell_name] = recs
            # Checkpoint-per-phase: persist the cell the moment it completes.
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / f"completions_{cell_name}.json").write_text(json.dumps(recs, indent=2))
            log.info("Cell %s persisted (%d records).", cell_name, len(recs))
    finally:
        _teardown_vllm(llm)
    return out


# ── Slot-stats subprocess (vLLM-free; 4 floats x trained AND base) ──────────


def run_slot_stats_subprocess(*, manifest_path: Path, log_path: Path) -> None:
    """Spawn the vLLM-free worker (vLLM monkey-patches transformers in-process;
    HF loads after an engine teardown are unreliable — parent rig pattern)."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--slot-stats-worker",
        "--manifest",
        str(manifest_path),
    ]
    log.info("Spawning slot-stats subprocess (manifest=%s log=%s)", manifest_path, log_path)
    env = {**os.environ}
    with log_path.open("ab") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = ""
        try:
            with log_path.open("rb") as f:
                f.seek(max(0, log_path.stat().st_size - 4096))
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            pass
        raise RuntimeError(
            f"Slot-stats subprocess failed (rc={proc.returncode}); log tail:\n{tail}"
        )


def _slot_stats_worker_main(*, manifest_path: Path) -> int:
    """Load base ONCE + adapter on top; per cell compute the 4-float slot stats
    on the TRAINED side (adapter enabled) and the BASE side (disable_adapter()),
    writing slot_stats_<cell>.json the moment the cell completes."""
    manifest = json.loads(manifest_path.read_text())
    adapter_dir = Path(manifest["adapter_dir"])
    cells = manifest["cells"]

    import gc

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats
    from explore_persona_space.train.sft import _pick_attn_implementation

    # Gauge assert BEFORE any logit readout (storage/analysis contract).
    assert_adapter_gauge_free(adapter_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=_pick_attn_implementation(),
        token=os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.eval()

    for cell in cells:
        cell_name = cell["name"]
        records = json.loads(Path(cell["records_in"]).read_text())
        contexts: list[str] = []
        for r in records:
            comp_clean = r["completion_text"].rstrip()
            if comp_clean.endswith(MARKER_TEXT.strip()):
                comp_clean = comp_clean[: -len(MARKER_TEXT.strip())].rstrip()
            contexts.append(r["prefix"] + comp_clean)
        log.info("Slot stats: cell=%s n=%d (trained side)", cell_name, len(contexts))
        trained = compute_marker_slot_stats(
            model,
            tokenizer,
            contexts=contexts,
            marker_text=MARKER_TEXT,
            position="end_of_answer",
            batch_size=LOGPROB_BATCH_SIZE,
            device="cuda:0",
            eos_token_id=EOS_TOKEN_ID,
        )
        log.info("Slot stats: cell=%s (base side via disable_adapter)", cell_name)
        with model.disable_adapter():
            based = compute_marker_slot_stats(
                model,
                tokenizer,
                contexts=contexts,
                marker_text=MARKER_TEXT,
                position="end_of_answer",
                batch_size=LOGPROB_BATCH_SIZE,
                device="cuda:0",
                eos_token_id=EOS_TOKEN_ID,
            )
        for side in (trained, based):
            for row in side:
                if not all(math.isfinite(v) for v in row.values()):
                    raise RuntimeError(f"Non-finite slot stat in cell={cell_name}: {row}")
        out_path = Path(cell["out"])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "adapter_dir": str(adapter_dir),
                    "n": len(contexts),
                    "trained": trained,
                    "base": based,
                }
            )
        )
        log.info("Slot stats persisted -> %s", out_path)

    del model, base
    gc.collect()
    torch.cuda.empty_cache()
    return 0


# ── Summary (verbatim parent) ────────────────────────────────────────────────


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def summarize_cell(records: list[dict], slot_stats: dict | None) -> dict:
    """Per-cell rollup: emission + truncation rates and the THREE-space slot
    means (log-prob PRIMARY, EOS-margin logit SECONDARY, probability sanity)."""
    n = len(records)
    summary = {
        "n": n,
        "emission_rate": sum(r["contains_marker"] for r in records) / max(n, 1),
        "ends_with_marker_rate": sum(r["ends_with_marker"] for r in records) / max(n, 1),
        "truncation_rate": sum(r["truncated"] for r in records) / max(n, 1),
    }
    if slot_stats is not None:
        tr, ba = slot_stats["trained"], slot_stats["base"]
        d_logp = [t["logp"] - b["logp"] for t, b in zip(tr, ba, strict=True)]
        d_zm = [t["z_marker"] - b["z_marker"] for t, b in zip(tr, ba, strict=True)]
        d_margin = [
            (t["z_marker"] - t["z_eos"]) - (b["z_marker"] - b["z_eos"])
            for t, b in zip(tr, ba, strict=True)
        ]
        summary.update(
            {
                "logp_trained_mean": _mean([t["logp"] for t in tr]),
                "logp_base_mean": _mean([b["logp"] for b in ba]),
                "delta_logp_mean": _mean(d_logp),
                "delta_z_marker_mean": _mean(d_zm),
                "delta_eos_margin_mean": _mean(d_margin),
                "logZ_trained_mean": _mean([t["logZ"] for t in tr]),
                "logZ_base_mean": _mean([b["logZ"] for b in ba]),
                "prob_trained_mean": _mean([math.exp(t["logp"]) for t in tr]),
                "prob_base_mean": _mean([math.exp(b["logp"]) for b in ba]),
            }
        )
    return summary


# ── Pod-side result-reporting (poll_pipeline.py contract, issue 558) ────────


def write_sentinel_558(slug: str, *, kind: str, note: str, version: int = 1) -> Path:
    """poll_pipeline-conformant sentinel for THIS issue (the pinned common's
    write_sentinel bakes in ISSUE=543 in both filename and task_id — reusing
    it would post markers onto task #543)."""
    d = sentinel_dir()
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"issue-{ISSUE_558}-{slug}-{int(time.time())}.json"
    payload = {
        "sentinel_schema_version": 1,
        "task_id": ISSUE_558,
        "kind": kind,
        "version": version,
        "gate": None,
        "blocks_pipeline": False,
        "by": "eval_issue558_panel",
        "ts": datetime.datetime.now(datetime.UTC).isoformat(),
        "note": note,
    }
    path.write_text(json.dumps(payload, indent=2))
    log.info("Sentinel written: %s (kind=%s)", path, kind)
    return path


# ── CPU-only launch-validity dry run (no vLLM / CUDA import) ─────────────────


def run_dry_run_cells(args: argparse.Namespace) -> int:
    """Build all 5 cells at full n + verify the launch-critical contracts on CPU.

    Covers: marker preflight (real tokenizer), eval-question fetch + count
    assert, cell construction (shapes / prompts / slugs digests), Hub
    adapter_config.json fetch + gauge assert for the requested adapter, and
    the parent-rollup anchor-reference lookup. Exits 0 with [phase=done]
    WITHOUT importing vLLM or touching CUDA.
    """
    phase_log("dry_run_cells")
    preflight = marker_preflight()
    eval_qs = ensure_eval_questions_local()
    cells = build_cells_panel(eval_qs, smoke=args.smoke)
    n_expect = N_SMOKE_PROMPTS if args.smoke else N_PANEL_PROMPTS

    digest: dict[str, Any] = {"cells": {}}
    for cell_name, items in cells.items():
        if len(items) != n_expect:
            raise RuntimeError(f"Cell {cell_name}: {len(items)} prompts, expected {n_expect}")
        persona_key = PANEL_CELLS[cell_name]
        if any(it["persona_key"] != persona_key for it in items):
            raise RuntimeError(f"Cell {cell_name}: persona_key mismatch vs PANEL_CELLS")
        if any(not it["user"].startswith(trigger_user("").rstrip()) for it in items):
            raise RuntimeError(f"Cell {cell_name}: trigger key missing from a user turn")
        digest["cells"][cell_name] = {
            "n": len(items),
            "persona_key": persona_key,
            "system_head": items[0]["system"][:60],
            "user_head": items[0]["user"][:80],
        }
    # All 5 cells share the identical question list (paired-contrast invariant).
    base_users = [it["user"] for it in cells["trigger50"]]
    for cell_name, items in cells.items():
        if [it["user"] for it in items] != base_users:
            raise RuntimeError(f"Cell {cell_name}: question list differs from trigger50")

    # Hub adapter_config gauge assert (CPU; the real artifact this run will load).
    from huggingface_hub import hf_hub_download

    from explore_persona_space.eval.marker_logprob import assert_gauge_free_adapter_config

    sub = f"adapters/{adapter_subfolder(args.arm, args.seed, PANEL_PHASE)}"
    cfg_path = hf_hub_download(
        repo_id=HUB_MODEL_REPO,
        filename=f"{sub}/adapter_config.json",
        token=os.environ.get("HF_TOKEN"),
    )
    cfg = json.loads(Path(cfg_path).read_text())
    assert_gauge_free_adapter_config(cfg, context=sub)
    digest["adapter_config"] = {"hub_subfolder": sub, "r": cfg.get("r"), "gauge_free": True}

    # Anchor-reference lookup (verifies the parent rollup keys exist pre-launch).
    ref = parent_doctor_reference(args.arm, args.seed)
    digest["anchor_reference"] = {
        "delta_logp_mean": ref["delta_logp_mean"],
        "delta_eos_margin_mean": ref["delta_eos_margin_mean"],
    }
    digest["marker_preflight"] = preflight

    out_dir = EVAL_RESULTS_DIR_558 / "dry_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"dry_run_cells_{args.arm}_seed{args.seed}.json"
    out_path.write_text(
        json.dumps(
            {**repro_metadata(), "issue": ISSUE_558, "mode": "dry_run_cells", **digest}, indent=2
        )
    )
    log.info("Dry run digest -> %s", out_path)
    phase_log("done")
    return 0


# ── Main entrypoint ──────────────────────────────────────────────────────────


def run_one(args: argparse.Namespace) -> int:
    phase_log("eval_gen")
    marker_preflight()
    adapter_dir = resolve_adapter(args.arm, args.seed, PANEL_PHASE, args.adapter_path)
    # Gauge assert up front too (cheap; the worker re-asserts before logit reads).
    assert_adapter_gauge_free(adapter_dir)
    eval_qs = ensure_eval_questions_local()
    cells = build_cells_panel(eval_qs, smoke=args.smoke)

    out_dir = EVAL_RESULTS_DIR_558 / args.arm / f"seed{args.seed}" / PANEL_PHASE
    if args.smoke:
        out_dir = out_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    lora_name = f"issue558_{cell_slug(args.arm, args.seed, PANEL_PHASE)}"

    records = generate_completions(
        adapter_dir=adapter_dir, lora_name=lora_name, cells=cells, out_dir=out_dir
    )

    phase_log("eval_slot_stats")
    manifest = {
        "adapter_dir": str(adapter_dir),
        "base_model": BASE_MODEL,
        "marker": MARKER_TEXT,
        "cells": [
            {
                "name": c,
                "records_in": str(out_dir / f"completions_{c}.json"),
                "out": str(out_dir / f"slot_stats_{c}.json"),
            }
            for c in cells
        ],
    }
    manifest_path = out_dir / "slot_stats_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    run_slot_stats_subprocess(manifest_path=manifest_path, log_path=out_dir / "slot_worker.log")

    summary = {
        **repro_metadata(),
        "issue": ISSUE_558,
        "parent_issue": 543,
        "arm": args.arm,
        "seed": args.seed,
        "phase": PANEL_PHASE,
        "smoke": args.smoke,
        "adapter_dir": str(adapter_dir),
        "adapter_hf_subfolder": f"adapters/{adapter_subfolder(args.arm, args.seed, PANEL_PHASE)}",
        "lora_id": lora_name,
        "max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "cells": {},
    }
    for c, recs in records.items():
        slot_path = out_dir / f"slot_stats_{c}.json"
        slot = json.loads(slot_path.read_text()) if slot_path.exists() else None
        summary["cells"][c] = summarize_cell(recs, slot)

    phase_log("anchor_check")
    if args.skip_anchor:
        log.warning("--skip-anchor: anchor check SKIPPED (diagnostic re-run only).")
        summary["anchor"] = {"anchor_checked": False, "reason": "--skip-anchor"}
    elif args.smoke:
        # Reduced-n smoke uses questions [0:20] (a strict subset of the
        # parent doctor cell's [0:50]) — the 50-prompt-mean comparison is
        # invalid by construction, so the anchor never gates here.
        log.warning("Smoke mode (reduced n): anchor check skipped (means not comparable).")
        summary["anchor"] = {"anchor_checked": False, "reason": "smoke-reduced-n"}
    else:
        summary["anchor"] = check_doctor_anchor(
            args.arm, args.seed, summary["cells"]["doctor"], gate=args.anchor_gate
        )

    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("run_summary -> %s", out_dir / "run_summary.json")

    if not args.skip_upload and not args.smoke:
        phase_log("eval_upload")
        from explore_persona_space.orchestrate.hub import upload_dataset_directory

        dest = f"{HUB_RAW_COMPLETIONS_BUCKET_558}/{cell_slug(args.arm, args.seed, PANEL_PHASE)}"
        upload_dataset_directory(out_dir, dest, pattern="completions_*.json")

    write_sentinel_558(
        f"eval-{args.arm}-seed{args.seed}",
        kind="epm:progress",
        note=json.dumps(
            {
                "event": "adapter_eval_complete",
                "arm": args.arm,
                "seed": args.seed,
                "anchor": summary["anchor"],
                "cells": {
                    c: {
                        k: summary["cells"][c].get(k)
                        for k in ("n", "emission_rate", "delta_logp_mean", "delta_eos_margin_mean")
                    }
                    for c in summary["cells"]
                },
            }
        ),
    )
    phase_log("done")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #558 persona-panel eval: 5 cells x 4-float marker slot stats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--slot-stats-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--manifest", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adapter-path", type=str, default=None, help="Local adapter dir (else Hub).")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="20 prompts/cell instead of 50.")
    p.add_argument(
        "--dry-run-cells",
        action="store_true",
        help="CPU-only launch-validity dry run (no vLLM/CUDA); exits after digests.",
    )
    p.add_argument(
        "--anchor-gate",
        action="store_true",
        help="Hard-FAIL on doctor-anchor breach (smoke/gate adapter r50_seed42 only).",
    )
    p.add_argument(
        "--skip-anchor",
        action="store_true",
        help="Skip the anchor check entirely (diagnostic re-runs only).",
    )
    p.add_argument("--skip-upload", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.slot_stats_worker:
        # Worker inherits the parent's CUDA_VISIBLE_DEVICES via the explicit
        # env passthrough; do NOT re-pin here.
        return _slot_stats_worker_main(manifest_path=Path(args.manifest))
    if args.arm is None:
        raise SystemExit("--arm is required")
    if args.dry_run_cells:
        return run_dry_run_cells(args)
    # Pin BEFORE any torch/vllm import touches CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    return run_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
