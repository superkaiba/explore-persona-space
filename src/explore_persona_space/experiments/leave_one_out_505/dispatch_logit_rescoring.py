# ruff: noqa: RUF002  # em-dash + minus sign + Qwen marker " ※" + Greek Δ intentional
"""Task #505 follow-up ``logit-space-rescoring`` — unified smoke=sweep dispatcher.

One dispatcher, one code path. The pod sweep is ``--cells 8 --seeds 3`` (24
adapters); the pod smoke is the SAME invocation with ``--cells 1 --seeds 1
--personas 2 --questions 2``; the local CPU smoke additionally swaps the
GPU-bound vLLM generation backend for ``--backend hf`` with a tiny stand-in
model (``--base-model Qwen/Qwen2.5-0.5B-Instruct --adapter-dir <throwaway>``).
Same per-adapter loop, same ``[phase=...]`` logging surface, same end-of-run
sentinel, same teardown sequence in every mode.

Phases (checkpoint-per-phase: every per-cell artifact is written the moment
it completes; ``EPM_SKIP_EXISTING`` resume is ON by default — re-running
skips cells whose artifact already exists, so a crash resumes where it died):

  gen          Phase A — vLLM on-policy generation + DV-A rescore per adapter
               (ONE engine, per-request LoRA hot-swap with unique int ids),
               per-cell ``raw_completions/<cell>_seed<S>.json`` + verified HF
               upload, then ONE hard vLLM teardown.
  capture      Phase B — four-float HF logit capture per adapter (ONE shared
               base model, PEFT load_adapter/set_adapter hot-swap; base side
               via ``disable_adapter()``), per-cell
               ``slot_stats/<cell>_seed<S>.json``.
  faithfulness Phase C — per-cell MAE + Spearman vs the stored frac-1.0
               trajectory values → ``faithfulness.json`` (re-written after
               every cell).
  done         sentinel write + terminal ``[phase=done]`` line.

Pod-side contract: this dispatcher NEVER shells out to ``scripts/task.py``
(branch-guarded to main); results reach the orchestrator through the
``poll_pipeline.py`` sentinel file (``/workspace/logs/issue-505-epm_results-
<epoch>.json`` with ``sentinel_schema_version`` / ``kind`` / ``version``) and
the ``[phase=done]`` log line.

Launch command (pod, 1× H100; ~3 GPU-h):

    nohup uv run python -m \\
      explore_persona_space.experiments.leave_one_out_505.dispatch_logit_rescoring \\
      --cells 8 --seeds 3 \\
      > /workspace/logs/issue-505-logit-rescoring-$(date +%s).log 2>&1 &
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    BASE_MODEL,
    HF_DATA_REPO,
    MARKER_TEXT,
)
from explore_persona_space.experiments.leave_one_out_505 import (
    CELL_SPECS,
    MAX_MODEL_LEN,
    MAX_NEW_TOKENS_GEN,
    SEEDS,
    SOURCE_PERSONA,
)
from explore_persona_space.experiments.leave_one_out_505.logit_rescoring import (
    RAW_COMPLETIONS_HF_PREFIX,
    SCHEMA_VERSION,
    TARGET_FRAC,
    _with_retries,
    assert_adapters_gauge_free,
    assert_generation_params_match_original,
    assert_real_marker_tokenization,
    capture_slot_stats_for_cell,
    download_adapter,
    expected_adapter_cells,
    faithfulness_for_cell,
    repro_block,
    resolve_adapter_repo_dirs,
    resolve_runtime_token_ids,
    run_phase_a_hf,
    run_phase_a_vllm,
    upload_raw_completions_file,
)
from explore_persona_space.utils import save_json_atomic

log = logging.getLogger("issue_505.dispatch_logit_rescoring")

DEFAULT_OUTPUT_ROOT = Path("eval_results/issue_505/logit-space-rescoring")
DEFAULT_SWEEP_DIR = Path("eval_results/issue_505/sweep")
SENTINEL_SCHEMA_VERSION = 1  # lockstep with poll_pipeline.SENTINEL_SCHEMA_VERSION_SUPPORTED


@dataclass
class RunContext:
    """Resolved inputs shared by the phase runners (built once by ``_prepare``)."""

    args: argparse.Namespace
    device: str
    dtype: Any
    out_root: Path
    sweep_dir: Path
    sentinel_dir: Path
    production: bool
    skip_existing: bool
    tokenizer: Any
    marker_id: int
    eos_id: int
    panel: list[str]
    questions: list[str]
    source: str
    personas: dict[str, str]
    cells: list[tuple[str, int]]
    adapter_dirs: dict[tuple[str, int], Path]
    base_inputs: dict[str, str]
    generation_params: dict[str, Any]
    uploaded: set[str] = field(default_factory=set)

    @property
    def gen_dir(self) -> Path:
        return self.out_root / "raw_completions"

    @property
    def stats_dir(self) -> Path:
        return self.out_root / "slot_stats"

    def gen_path(self, slug: str, seed: int) -> Path:
        return self.gen_dir / f"{slug}_seed{seed}.json"

    def stats_path(self, slug: str, seed: int) -> Path:
        return self.stats_dir / f"{slug}_seed{seed}.json"


def _resolve_panel_and_questions(
    sweep_dir: Path, *, personas_cap: int | None, questions_cap: int | None
) -> tuple[list[str], list[str], str]:
    """Panel persona names + eval questions + source, read from a STORED trajectory.

    Reading the original sweep's own artifact (rather than re-deriving the
    panel) guarantees identity with the original eval grid. Caps apply to the
    held-out panel only; the source persona is ALWAYS appended (the source
    covariate + faithfulness anchor need it).
    """
    ref = sweep_dir / "c505_full_set" / f"seed_{SEEDS[0]}" / "trajectory.json"
    if not ref.exists():
        raise FileNotFoundError(
            f"reference trajectory missing at {ref} — the stored #505 sweep artifacts are "
            "required to pin the panel/questions (and for Phase C). Run from the repo root."
        )
    payload = json.loads(ref.read_text())
    held_out = list(payload["held_out_personas"])
    questions = list(payload["eval_questions"])
    source = str(payload["source"])
    if personas_cap is not None:
        held_out = held_out[:personas_cap]
    if questions_cap is not None:
        questions = questions[:questions_cap]
    return held_out, questions, source


def _load_persona_prompts(panel: list[str], source: str) -> dict[str, str]:
    """{persona: system_prompt} for panel + source from the inherited #472 bank."""
    from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
        load_persona_bank,
    )
    from explore_persona_space.experiments.leave_one_out_505.dispatch import (
        _l10_centroid_dir,
        _prefetch_inherited_artifacts,
    )

    i472_root = _l10_centroid_dir()
    _prefetch_inherited_artifacts(i472_root)
    bank = load_persona_bank(i472_root / "persona_bank.json")
    missing = [p for p in [*panel, source] if p not in bank]
    if missing:
        raise KeyError(f"persona bank missing prompts for {missing} — wrong bank artifact?")
    personas = {p: bank[p] for p in panel}
    personas[source] = bank[source]  # source LAST, always present
    return personas


def _prepare(args: argparse.Namespace) -> RunContext:
    """Launch-blocking invariants + input/adapter resolution + manifest write."""
    import torch

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    out_root = Path(args.output_root)
    sweep_dir = Path(args.sweep_dir)
    sentinel_dir = (
        Path(args.sentinel_dir)
        if args.sentinel_dir
        else (Path("/workspace/logs") if Path("/workspace").exists() else out_root / "logs")
    )
    production = (
        args.backend == "vllm"
        and args.base_model == BASE_MODEL
        and args.adapter_dir is None
        and args.max_new_tokens == MAX_NEW_TOKENS_GEN
    )
    skip_existing = (
        os.environ.get("EPM_SKIP_EXISTING", "1").lower() in {"1", "true", "yes"}
        and not args.force_regen
    )
    log.info(
        "[phase=start] backend=%s cells=%d seeds=%d personas=%s questions=%s device=%s "
        "production=%s skip_existing=%s host=%s",
        args.backend,
        args.cells,
        args.seeds,
        args.personas,
        args.questions,
        device,
        production,
        skip_existing,
        socket.gethostname(),
    )

    # ── Launch-blocking invariants (all BEFORE any GPU work) ────────────────
    log.info("[phase=invariants] marker tokenization + generation params")
    assert_real_marker_tokenization()  # canonical 7B tokenizer, always
    if production:
        assert_generation_params_match_original(
            max_new_tokens=args.max_new_tokens, max_model_len=MAX_MODEL_LEN
        )
    else:
        log.warning(
            "[smoke-mode] non-production overrides active (backend=%s base_model=%s "
            "adapter_dir=%s max_new_tokens=%d) — results are wiring-only, NOT science.",
            args.backend,
            args.base_model,
            args.adapter_dir,
            args.max_new_tokens,
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    marker_id, eos_id = resolve_runtime_token_ids(tokenizer, production=production)

    log.info("[phase=resolve_inputs] panel + questions from %s", sweep_dir)
    panel, questions, source = _resolve_panel_and_questions(
        sweep_dir, personas_cap=args.personas, questions_cap=args.questions
    )
    assert source == SOURCE_PERSONA, (source, SOURCE_PERSONA)
    personas = _load_persona_prompts(panel, source)
    log.info(
        "[phase=resolve_inputs] %d held-out + source=%s; %d questions",
        len(panel),
        source,
        len(questions),
    )

    cells = expected_adapter_cells(args.cells, args.seeds)
    log.info("[phase=resolve_adapters] %d (cell, seed) pairs", len(cells))
    adapter_dirs: dict[tuple[str, int], Path] = {}
    adapter_revision: str | None = None
    if args.adapter_dir is not None:
        # Smoke override: every cell points at the local throwaway adapter.
        adapter_dirs = {cs: Path(args.adapter_dir) for cs in cells}
    else:
        repo_dirs, adapter_revision = resolve_adapter_repo_dirs(cells)
        for cs in cells:
            adapter_dirs[cs] = download_adapter(repo_dirs[cs], revision=adapter_revision)
            log.info("[phase=resolve_adapters] %s_seed%d -> %s", cs[0], cs[1], adapter_dirs[cs])
    assert_adapters_gauge_free(adapter_dirs)

    base_inputs = {
        "sweep_dir": str(sweep_dir),
        "hf_data_repo": HF_DATA_REPO,
        "adapter_revision": adapter_revision or f"local:{args.adapter_dir}",
    }
    generation_params = {
        "temperature": 0.0,
        "top_p": 1.0,
        "n": 1,
        "max_new_tokens": args.max_new_tokens,
        "max_model_len": MAX_MODEL_LEN,
        "greedy": True,
    }
    ctx = RunContext(
        args=args,
        device=device,
        dtype=dtype,
        out_root=out_root,
        sweep_dir=sweep_dir,
        sentinel_dir=sentinel_dir,
        production=production,
        skip_existing=skip_existing,
        tokenizer=tokenizer,
        marker_id=marker_id,
        eos_id=eos_id,
        panel=panel,
        questions=questions,
        source=source,
        personas=personas,
        cells=cells,
        adapter_dirs=adapter_dirs,
        base_inputs=base_inputs,
        generation_params=generation_params,
    )
    ctx.gen_dir.mkdir(parents=True, exist_ok=True)
    ctx.stats_dir.mkdir(parents=True, exist_ok=True)
    save_json_atomic(
        out_root / "manifest.json",
        {
            "schema_version": SCHEMA_VERSION,
            "frac": TARGET_FRAC,
            "backend": args.backend,
            "base_model": args.base_model,
            "production_params": production,
            "generation_params": generation_params,
            "marker_text": MARKER_TEXT,
            "runtime_marker_id": marker_id,
            "runtime_eos_id": eos_id,
            "cells": [f"{s}_seed{sd}" for s, sd in cells],
            "n_held_out": len(panel),
            "source": source,
            "n_questions": len(questions),
            "skip_existing": skip_existing,
            "reproducibility": repro_block(base_inputs),
        },
    )
    log.info("[phase=manifest] %s", out_root / "manifest.json")

    # Which raw-completion files already landed on the hub (resume-safe upload).
    if not args.no_upload:
        from huggingface_hub import HfApi

        api = HfApi(token=os.environ.get("HF_TOKEN"))
        ctx.uploaded = {
            f
            for f in _with_retries(
                lambda: api.list_repo_files(HF_DATA_REPO, repo_type="dataset"),
                f"list_repo_files({HF_DATA_REPO})",
            )
            if f.startswith(RAW_COMPLETIONS_HF_PREFIX + "/")
        }
        log.info("[phase=gen] %d raw-completion files already on the hub", len(ctx.uploaded))
    return ctx


# ── Phase A persistence + upload helpers ─────────────────────────────────────


def _persist_phase_a(ctx: RunContext, slug: str, seed: int, payload: dict) -> None:
    save_json_atomic(
        ctx.gen_path(slug, seed),
        {
            "schema_version": SCHEMA_VERSION,
            "cell": slug,
            "seed": seed,
            "frac": TARGET_FRAC,
            "backend": ctx.args.backend,
            "base_model": ctx.args.base_model,
            "adapter_dir": str(ctx.adapter_dirs[(slug, seed)]),
            "personas": [*ctx.panel, ctx.source],
            "questions": ctx.questions,
            "generation_params": ctx.generation_params,
            **payload,
            "reproducibility": repro_block(ctx.base_inputs),
        },
    )


def _upload_phase_a(ctx: RunContext, slug: str, seed: int) -> None:
    path_in_repo = f"{RAW_COMPLETIONS_HF_PREFIX}/{slug}_seed{seed}.json"
    if ctx.args.no_upload or path_in_repo in ctx.uploaded:
        return
    upload_raw_completions_file(ctx.gen_path(slug, seed), cell_slug=slug, seed=seed)
    ctx.uploaded.add(path_in_repo)


def _run_phase_a_vllm_all(ctx: RunContext) -> None:
    """Phase A, vLLM backend: ONE engine, unique-int-id LoRA hot-swap, hard teardown."""
    from vllm import LLM

    llm = LLM(
        model=ctx.args.base_model,
        dtype="bfloat16",
        gpu_memory_utilization=ctx.args.gpu_mem_util,
        seed=SEEDS[0],
        max_model_len=MAX_MODEL_LEN,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
    )
    try:
        for idx, (slug, seed) in enumerate(ctx.cells):
            if ctx.skip_existing and ctx.gen_path(slug, seed).exists():
                log.info("[phase=gen] skip-existing %s_seed%d", slug, seed)
                _upload_phase_a(ctx, slug, seed)  # heal a crash between write + upload
                continue
            try:
                payload = run_phase_a_vllm(
                    llm,
                    ctx.tokenizer,
                    cell_slug=slug,
                    seed=seed,
                    adapter_dir=ctx.adapter_dirs[(slug, seed)],
                    lora_int_id=idx + 1,  # UNIQUE per adapter — vLLM caches by int id
                    personas=ctx.personas,
                    questions=ctx.questions,
                    max_new_tokens=ctx.args.max_new_tokens,
                )
                _persist_phase_a(ctx, slug, seed, payload)
                _upload_phase_a(ctx, slug, seed)
                log.info("[phase=gen] %s_seed%d done (%d/%d)", slug, seed, idx + 1, len(ctx.cells))
            except Exception:
                log.exception("[phase=gen] %s_seed%d FAILED", slug, seed)
                raise
    finally:
        from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
            _teardown_vllm_hard,
        )

        log.info("[phase=vllm_teardown] hard teardown (worker reap + cache clear)")
        _teardown_vllm_hard(llm)


def _hot_swap_adapter(peft_model, base, adapter_dir: Path, name: str, prev_name: str | None):
    """Load/set one adapter on the shared base (first call wraps the base)."""
    from peft import PeftModel

    if peft_model is None:
        peft_model = PeftModel.from_pretrained(base, str(adapter_dir), adapter_name=name).eval()
    else:
        peft_model.load_adapter(str(adapter_dir), adapter_name=name)
        peft_model.set_adapter(name)
        if prev_name is not None and prev_name != name:
            peft_model.delete_adapter(prev_name)
    return peft_model


def _load_base_model(ctx: RunContext):
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM.from_pretrained(
        ctx.args.base_model,
        torch_dtype=ctx.dtype,
        device_map={"": ctx.device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    ).eval()


def _run_phase_a_hf_all(ctx: RunContext) -> None:
    """Phase A, HF backend (CPU smoke): same loop shape, batched greedy generate."""
    base = _load_base_model(ctx)
    peft_model = None
    prev_name: str | None = None
    for idx, (slug, seed) in enumerate(ctx.cells):
        if ctx.skip_existing and ctx.gen_path(slug, seed).exists():
            log.info("[phase=gen] skip-existing %s_seed%d", slug, seed)
            _upload_phase_a(ctx, slug, seed)
            continue
        try:
            name = f"{slug}_s{seed}"
            peft_model = _hot_swap_adapter(
                peft_model, base, ctx.adapter_dirs[(slug, seed)], name, prev_name
            )
            prev_name = name
            payload = run_phase_a_hf(
                peft_model,
                ctx.tokenizer,
                personas=ctx.personas,
                questions=ctx.questions,
                max_new_tokens=ctx.args.max_new_tokens,
                device=ctx.device,
                batch_size=ctx.args.batch_size,
            )
            _persist_phase_a(ctx, slug, seed, payload)
            _upload_phase_a(ctx, slug, seed)
            log.info("[phase=gen] %s_seed%d done (%d/%d)", slug, seed, idx + 1, len(ctx.cells))
        except Exception:
            log.exception("[phase=gen] %s_seed%d FAILED", slug, seed)
            raise
    del peft_model, base


def _run_phase_b_all(ctx: RunContext) -> None:
    """Phase B: four-float HF capture, ONE shared base, PEFT hot-swap per cell."""
    log.info("[phase=capture] HF four-float capture over %d cells", len(ctx.cells))
    base = _load_base_model(ctx)
    peft_model = None
    prev_name: str | None = None
    equivalence_done = False
    equivalence_tol = 1e-3 if ctx.device == "cpu" else 0.25  # fp32 vs bf16 kernels
    for idx, (slug, seed) in enumerate(ctx.cells):
        stats_path = ctx.stats_path(slug, seed)
        if ctx.skip_existing and stats_path.exists():
            log.info("[phase=capture] skip-existing %s_seed%d", slug, seed)
            continue
        gen_path = ctx.gen_path(slug, seed)
        if not gen_path.exists():
            raise FileNotFoundError(
                f"Phase A artifact missing at {gen_path} for {slug}_seed{seed} — Phase B needs "
                "the regenerated completions. Re-run the dispatcher (Phase A resumes)."
            )
        try:
            phase_a = json.loads(gen_path.read_text())
            name = f"{slug}_s{seed}"
            peft_model = _hot_swap_adapter(
                peft_model, base, ctx.adapter_dirs[(slug, seed)], name, prev_name
            )
            prev_name = name
            stats = capture_slot_stats_for_cell(
                peft_model,
                ctx.tokenizer,
                completions=phase_a["completions"],
                personas=ctx.personas,
                questions=ctx.questions,
                marker_id=ctx.marker_id,
                eos_id=ctx.eos_id,
                device=ctx.device,
                batch_size=ctx.args.batch_size,
                run_equivalence_guard=not equivalence_done,
                equivalence_float_tol=equivalence_tol,
            )
            equivalence_done = True
            save_json_atomic(
                stats_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "cell": slug,
                    "seed": seed,
                    "frac": TARGET_FRAC,
                    "base_model": ctx.args.base_model,
                    "adapter_dir": str(ctx.adapter_dirs[(slug, seed)]),
                    "marker_id": ctx.marker_id,
                    "eos_id": ctx.eos_id,
                    "personas": [*ctx.panel, ctx.source],
                    "questions": ctx.questions,
                    "slot_stats": stats,
                    "reproducibility": repro_block(ctx.base_inputs),
                },
            )
            log.info("[phase=capture] %s_seed%d done (%d/%d)", slug, seed, idx + 1, len(ctx.cells))
        except Exception:
            log.exception("[phase=capture] %s_seed%d FAILED", slug, seed)
            raise
    del peft_model, base


def _run_phase_c_all(ctx: RunContext) -> dict[str, dict]:
    """Phase C: faithfulness vs the stored frac-1.0 trajectories (rewritten per cell)."""
    log.info("[phase=faithfulness] per-cell MAE + Spearman vs stored trajectories")
    faith_path = ctx.out_root / "faithfulness.json"
    per_cell: dict[str, dict] = {}
    for slug, seed in ctx.cells:
        phase_a = json.loads(ctx.gen_path(slug, seed).read_text())
        phase_b = json.loads(ctx.stats_path(slug, seed).read_text())
        stored_path = ctx.sweep_dir / slug / f"seed_{seed}" / "trajectory.json"
        per_cell[f"{slug}_seed{seed}"] = faithfulness_for_cell(
            cell_slug=slug,
            seed=seed,
            stored_trajectory_path=stored_path,
            phase_a_payload=phase_a,
            phase_b_stats=phase_b["slot_stats"],
            personas=[*ctx.panel, ctx.source],
            questions=ctx.questions,
        )
        # Checkpoint-per-phase: rewrite after EVERY cell (idempotent overwrite).
        save_json_atomic(
            faith_path,
            {
                "schema_version": SCHEMA_VERSION,
                "frac": TARGET_FRAC,
                "per_cell": per_cell,
                "n_cells_done": len(per_cell),
                "n_cells_expected": len(ctx.cells),
                "reproducibility": repro_block(ctx.base_inputs),
            },
        )
    log.info("[phase=faithfulness] wrote %s (%d cells)", faith_path, len(per_cell))
    return per_cell


def _write_sentinel(sentinel_dir: Path, *, note: str) -> Path:
    """End-of-run sentinel conforming to ``poll_pipeline._SENTINEL_REQUIRED_KEYS``."""
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    path = sentinel_dir / f"issue-505-epm_results-{int(time.time())}.json"
    save_json_atomic(
        path,
        {
            "sentinel_schema_version": SENTINEL_SCHEMA_VERSION,
            "kind": "epm:results",
            "version": 1,
            "task_id": 505,
            "by": "dispatch_logit_rescoring",
            "ts": repro_block({})["timestamp_utc"],
            "gate": "",
            "blocks_pipeline": False,
            "note": note,
        },
    )
    log.info("[sentinel] wrote %s", path)
    return path


def main(args: argparse.Namespace) -> int:
    """Run phases A → B → C on the requested (cells × seeds) grid. Returns exit code."""
    # uv run python does NOT auto-load .env; load at main()-top so HF_TOKEN /
    # WANDB keys exist before any hub call (also pins HF_HOME on pods).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    )
    ctx = _prepare(args)

    log.info("[phase=gen] backend=%s over %d cells", args.backend, len(ctx.cells))
    if args.backend == "vllm":
        _run_phase_a_vllm_all(ctx)
    else:
        _run_phase_a_hf_all(ctx)

    _run_phase_b_all(ctx)
    per_cell = _run_phase_c_all(ctx)

    hf_g = [c["hf_vs_stored"]["g"] for c in per_cell.values()]
    note = (
        f"logit-space-rescoring phases A-C complete: {len(ctx.cells)} cells at frac "
        f"{TARGET_FRAC}; backend={args.backend}; production_params={ctx.production}; "
        f"outputs under {ctx.out_root}; "
        f"hf_vs_stored g-side MAE range "
        f"[{min(x['mae'] for x in hf_g):.3f}, {max(x['mae'] for x in hf_g):.3f}] nats, "
        f"Spearman range [{min(x['spearman_rho'] for x in hf_g):.3f}, "
        f"{max(x['spearman_rho'] for x in hf_g):.3f}] across cells. "
        f"Phase D (analyze_logit_rescoring) runs on the VM."
    )
    _write_sentinel(ctx.sentinel_dir, note=note)
    log.info("[phase=done] all %d cells finished", len(ctx.cells))
    return 0


def cli_main(argv: list[str] | None = None) -> int:
    """argparse entrypoint (mirrors the sibling #505 dispatcher's CLI shape)."""
    p = argparse.ArgumentParser(description="#505 logit-space-rescoring smoke=sweep dispatcher")
    p.add_argument("--cells", type=int, default=len(CELL_SPECS), help="arms to run (cap; max 8)")
    p.add_argument("--seeds", type=int, default=len(SEEDS), help="seeds to run (cap; max 3)")
    p.add_argument("--personas", type=int, default=None, help="cap on held-out panel (smoke)")
    p.add_argument("--questions", type=int, default=None, help="cap on eval questions (smoke)")
    p.add_argument(
        "--backend",
        choices=("vllm", "hf"),
        default="vllm",
        help="generation backend; 'hf' is the CPU-smoke path (no vLLM rescore)",
    )
    p.add_argument("--base-model", default=BASE_MODEL, help="runtime model (smoke: 0.5B stand-in)")
    p.add_argument(
        "--adapter-dir",
        default=None,
        help="local adapter dir override for ALL cells (smoke throwaway); default: HF download",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=MAX_NEW_TOKENS_GEN,
        help="generation cap; production asserts equality with the original eval's constant",
    )
    p.add_argument("--batch-size", type=int, default=8, help="HF forward/generate batch size")
    p.add_argument("--gpu-mem-util", type=float, default=0.85, help="vLLM gpu_memory_utilization")
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    p.add_argument("--sweep-dir", default=str(DEFAULT_SWEEP_DIR))
    p.add_argument("--sentinel-dir", default=None, help="default: /workspace/logs on pods")
    p.add_argument(
        "--no-upload", action="store_true", help="skip HF raw-completion uploads (local smoke only)"
    )
    p.add_argument(
        "--force-regen",
        action="store_true",
        help="ignore existing per-cell artifacts (default resumes via EPM_SKIP_EXISTING=1)",
    )
    args = p.parse_args(argv)
    return main(args)


if __name__ == "__main__":
    sys.exit(cli_main(sys.argv[1:]))
