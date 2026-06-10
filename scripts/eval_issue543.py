#!/usr/bin/env python3
"""Issue #543 eval — on-policy emission + 4-float marker slot stats per cell.

Per (arm, seed, phase): vLLM greedy generation over 4 cells, then a FRESH
subprocess (never imports vLLM) computes the marker slot stats — the FOUR
floats per slot per model side (log P, z_marker, z_eos, logZ; storage
contract of .claude/rules/marker-leakage-measurement.md) — on the trained
adapter AND the base model (same process, ``disable_adapter()``), persisting
per-cell JSON the moment each cell completes.

Changes vs the #506 eval this adapts (plan §4.4): Qwen-2.5-7B constants;
full-vocab KL DELETED (banned, #504); adapter served via vLLM
``enable_lora``/``LoRARequest`` (r=16 attn-only, no merged dirs); ONE FRESH
vLLM ENGINE PER ADAPTER (one adapter per invocation), adapter path + LoRA id
logged in every record; ``assert_gauge_free_adapter_config`` before any
logit read.

Cells (plan §10; eval questions = the chain's 250 held-out, DETERMINISTIC
slices, no shuffle): trigger = assistant+key on [0:200]; no_trigger =
assistant on the same [0:200]; doctor = medical_doctor+key on [0:50];
reference = assistant, no key, on [200:250].

Usage (pod, 1 GPU):
    uv run python scripts/eval_issue543.py --arm r50 --seed 42 --phase phase1 \\
        --adapter-path /path/to/adapter --gpu 0
    uv run python scripts/eval_issue543.py --arm r50 --seed 42 --phase phase1 --smoke ...
Dev-check mode (plan §4.2.6 — 50 greedy gens at assistant+key on [200:250]):
    uv run python scripts/eval_issue543.py --dev-check --adapter-path ... --gpu 0 \\
        --out /path/dev_check.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="eval_issue543")

import os  # noqa: E402

from _issue543_common import (  # noqa: E402
    ARMS,
    BASE_MODEL,
    DEFAULT_ASSISTANT_KEY,
    DEV_CHECK_MIN_EMIT,
    DEV_CHECK_N,
    EOS_TOKEN_ID,
    EVAL_MAX_NEW_TOKENS,
    EVAL_RESULTS_DIR,
    HUB_MODEL_REPO,
    HUB_RAW_COMPLETIONS_BUCKET,
    HUB_RAW_COMPLETIONS_BUCKET_557,
    MARKER_TEXT,
    N_DOCTOR_PROMPTS,
    N_REFERENCE_PROMPTS,
    N_SMOKE_PROMPTS,
    N_TRIGGER_PROMPTS,
    PHASES,
    adapter_subfolder_v,
    all_persona_prompts,
    cell_slug,
    cell_slug_v,
    ensure_eval_questions_local,
    marker_preflight,
    phase_log,
    repro_metadata,
    trigger_user,
    truncated,
    validate_variant,
    variant_cell_dir,
)

log = logging.getLogger("eval_issue543")

LOGPROB_BATCH_SIZE = 8


# ── Cells (deterministic slices; plan §10) ──────────────────────────────────


def build_cells(eval_questions: list[str], *, smoke: bool) -> dict[str, list[dict]]:
    """4 eval cells over the chain's 250 held-out questions (no shuffle)."""
    personas = all_persona_prompts()
    asst = personas[DEFAULT_ASSISTANT_KEY]
    doctor = personas["medical_doctor"]

    n_t = N_SMOKE_PROMPTS if smoke else N_TRIGGER_PROMPTS
    n_doc = N_SMOKE_PROMPTS if smoke else N_DOCTOR_PROMPTS
    n_ref = N_SMOKE_PROMPTS if smoke else N_REFERENCE_PROMPTS
    t_qs = eval_questions[:n_t]
    ref_qs = eval_questions[200 : 200 + n_ref]

    cells = {
        "trigger": [
            {"system": asst, "user": trigger_user(q), "persona_key": "assistant", "trigger": True}
            for q in t_qs
        ],
        "no_trigger": [
            {"system": asst, "user": q, "persona_key": "assistant", "trigger": False} for q in t_qs
        ],
        "doctor": [
            {
                "system": doctor,
                "user": trigger_user(q),
                "persona_key": "medical_doctor",
                "trigger": True,
            }
            for q in t_qs[:n_doc]
        ],
        "reference": [
            {"system": asst, "user": q, "persona_key": "assistant", "trigger": False}
            for q in ref_qs
        ],
    }
    empty = [k for k, v in cells.items() if not v]
    if empty:
        raise RuntimeError(f"Empty cell(s) (smoke={smoke}): {empty}")
    return cells


def build_dev_cell(eval_questions: list[str]) -> dict[str, list[dict]]:
    """Dev-check cell: assistant+key on questions [200:250] (disjoint from the
    200 final-eval trigger questions; plan §4.2.6)."""
    asst = all_persona_prompts()[DEFAULT_ASSISTANT_KEY]
    qs = eval_questions[200 : 200 + DEV_CHECK_N]
    if len(qs) != DEV_CHECK_N:
        raise RuntimeError(f"Dev-check needs {DEV_CHECK_N} questions; got {len(qs)}")
    return {
        "dev": [
            {"system": asst, "user": trigger_user(q), "persona_key": "assistant", "trigger": True}
            for q in qs
        ]
    }


# ── Adapter resolution ──────────────────────────────────────────────────────


def resolve_adapter(
    arm: str, seed: int, phase: str, adapter_path: str | None, variant: str | None = None
) -> Path:
    """Local path if given (dispatcher hand-off); else fetch from HF Hub.

    With ``variant`` set (#557 lr sweep) the Hub fallback resolves the
    ``adapters/issue557/...`` subfolder the variant cell uploaded to — the
    issue_543 subfolder would silently serve the WRONG (1e-4 anchor) adapter.
    """
    if adapter_path:
        p = Path(adapter_path)
        if not p.exists() or not (p / "adapter_config.json").exists():
            raise FileNotFoundError(f"--adapter-path invalid (no adapter_config.json): {p}")
        return p
    from huggingface_hub import snapshot_download

    sub = f"adapters/{adapter_subfolder_v(arm, seed, phase, variant)}"
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


# ── vLLM generation (FRESH engine per adapter) ──────────────────────────────


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

    Adapter isolation (round-1 critique fix, plan §4.4): the engine is created
    fresh in this invocation, serves exactly one LoRA, and every record logs
    the adapter path + LoRA id — sequential-adapter state contamination of the
    primary DV is ruled out by construction.
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
        # 0.70 (not the 0.9 default): eval children share the GPU with the
        # cell's training process, which can hold residual memory even after
        # cleanup. 0.70 x 79.2 GiB = 55.4 GiB (model ~15 GiB + ~40 GiB KV —
        # ample for max_num_seqs=64 x 4096 ctx); greedy outputs are unaffected
        # by KV-pool size. 2026-06-10 smoke-cell incident: dev-check vLLM died
        # on the 0.9 startup check with 63.4 GiB free.
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
    HF loads after an engine teardown are unreliable — same pattern as #475/#506)."""
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


# ── Summary ──────────────────────────────────────────────────────────────────


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


# ── Entrypoints ──────────────────────────────────────────────────────────────


def run_dev_check(args: argparse.Namespace) -> int:
    """Post-stop manipulation check (plan §4.2.6): 50 greedy completions at
    assistant+key on questions [200:250]; PASS iff >= 48/50 emit the marker."""
    adapter_dir = resolve_adapter(args.arm, args.seed, "phase1", args.adapter_path)
    eval_qs = ensure_eval_questions_local()
    cells = build_dev_cell(eval_qs)
    out_dir = Path(args.out).parent if args.out else EVAL_RESULTS_DIR / "dev_checks"
    out_dir.mkdir(parents=True, exist_ok=True)
    recs = generate_completions(
        adapter_dir=adapter_dir,
        lora_name=f"issue543_dev_{cell_slug(args.arm, args.seed, 'phase1')}",
        cells=cells,
        out_dir=out_dir,
    )["dev"]
    n_emit = sum(r["contains_marker"] for r in recs)
    result = {
        **repro_metadata(),
        "mode": "dev_check",
        "arm": args.arm,
        "seed": args.seed,
        "adapter_dir": str(adapter_dir),
        "n": len(recs),
        "n_emit": n_emit,
        "threshold": DEV_CHECK_MIN_EMIT,
        "passed": n_emit >= DEV_CHECK_MIN_EMIT,
    }
    out = Path(args.out) if args.out else out_dir / f"dev_check_{args.arm}_seed{args.seed}.json"
    out.write_text(json.dumps(result, indent=2))
    log.info("Dev check: %d/%d emitted (pass=%s) -> %s", n_emit, len(recs), result["passed"], out)
    phase_log("done")
    return 0


def run_one(args: argparse.Namespace) -> int:
    phase_log("eval_gen")
    marker_preflight()
    variant = args.variant
    adapter_dir = resolve_adapter(args.arm, args.seed, args.phase, args.adapter_path, variant)
    # Gauge assert up front too (cheap; the worker re-asserts before logit reads).
    assert_adapter_gauge_free(adapter_dir)
    eval_qs = ensure_eval_questions_local()
    cells = build_cells(eval_qs, smoke=args.smoke)

    if variant is None:
        out_dir = EVAL_RESULTS_DIR / args.arm / f"seed{args.seed}" / args.phase
    else:
        # #557 OUTPUT namespace: eval_results/issue_557/<arm>/<variant>/seed<S>/phase2
        out_dir = variant_cell_dir(args.arm, variant, args.seed) / args.phase
    if args.smoke:
        out_dir = out_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "issue543" if variant is None else "issue557"
    lora_name = f"{prefix}_{cell_slug_v(args.arm, args.seed, args.phase, variant)}"

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
        "arm": args.arm,
        "seed": args.seed,
        "phase": args.phase,
        "variant": variant,
        "smoke": args.smoke,
        "adapter_dir": str(adapter_dir),
        "adapter_hf_subfolder": (
            f"adapters/{adapter_subfolder_v(args.arm, args.seed, args.phase, variant)}"
        ),
        "lora_id": lora_name,
        "max_new_tokens": EVAL_MAX_NEW_TOKENS,
        "cells": {},
    }
    for c, recs in records.items():
        slot_path = out_dir / f"slot_stats_{c}.json"
        slot = json.loads(slot_path.read_text()) if slot_path.exists() else None
        summary["cells"][c] = summarize_cell(recs, slot)
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2))
    log.info("run_summary -> %s", out_dir / "run_summary.json")

    if not args.skip_upload and not args.smoke:
        phase_log("eval_upload")
        from explore_persona_space.orchestrate.hub import upload_dataset_directory

        bucket = HUB_RAW_COMPLETIONS_BUCKET if variant is None else HUB_RAW_COMPLETIONS_BUCKET_557
        dest = f"{bucket}/{cell_slug_v(args.arm, args.seed, args.phase, variant)}"
        upload_dataset_directory(out_dir, dest, pattern="completions_*.json")

    phase_log("done")
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #543 eval: on-policy emission + 4-float marker slot stats.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--slot-stats-worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--manifest", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--arm", choices=ARMS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--phase", choices=PHASES, default="phase1")
    p.add_argument("--adapter-path", type=str, default=None, help="Local adapter dir (else Hub).")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--smoke", action="store_true", help="20 prompts/cell instead of 200/50.")
    p.add_argument("--dev-check", action="store_true", help="Post-stop manipulation check mode.")
    p.add_argument("--out", type=str, default=None, help="Output JSON path (dev-check mode).")
    p.add_argument("--skip-upload", action="store_true")
    p.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Issue #557 lr tag (e.g. lr3e5): routes out_dir + raw-completion "
        "bucket + Hub adapter fallback to issue_557 namespaces. Requires "
        "--phase phase2; invalid with --dev-check.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.slot_stats_worker:
        # Worker inherits the parent's CUDA_VISIBLE_DEVICES via the explicit
        # env passthrough; do NOT re-pin here.
        return _slot_stats_worker_main(manifest_path=Path(args.manifest))
    if args.variant is not None:
        validate_variant(args.variant)
        if args.dev_check:
            raise SystemExit("--variant is invalid with --dev-check (a Phase-1-only mode).")
        if args.phase != "phase2":
            raise SystemExit(
                "--variant requires --phase phase2 (the #557 sweep reuses the "
                "parent's Phase-1 evals; only post-SFT cells are variant-namespaced)."
            )
    # Pin BEFORE any torch/vllm import touches CUDA.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.dev_check:
        if args.arm is None:
            raise SystemExit("--dev-check requires --arm")
        return run_dev_check(args)
    if args.arm is None:
        raise SystemExit("--arm is required")
    return run_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
