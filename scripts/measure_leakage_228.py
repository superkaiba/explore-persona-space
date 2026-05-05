#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Issue #228 Phase 0.5 — single-state leakage measurement at 11 targets.

For ONE (source, checkpoint_step) state:

  1. Idempotency: if
     ``eval_results/causal_proximity/strong_convergence/<source>/checkpoint-<step>/marker_eval.json``
     exists, exit 0.
  2. Determine which adapters to merge:
       * ``checkpoint_step > 0``: merge convergence adapter from
         ``ADAPTER_MAP[source]/checkpoint-<step>`` AND marker LoRA at
         ``adapters/cp_marker_<source>_ep<step>_s42`` (must already be on
         HF Hub — trained by Phase 0).
       * ``checkpoint_step == 0``: NO convergence adapter (epoch-0 baseline).
         Just the marker LoRA at ``adapters/cp_marker_<source>_ep0_s42``
         (already on HF Hub for all 7 sources).
  3. vLLM batched generation: 11 personas × 20 questions × 10 completions
     @ temp=1.0, top_p=1.0, max_tokens=512, seed=42 — bit-identical to
     #109's `eval_causal_ckpt.py`.
  4. Per (target_persona, question), count completions whose decoded text
     contains ``[ZLT]`` (case-insensitive substring) — same as #109.
  5. Compute ``rate = hits / 200`` per target persona (200 = 20 questions
     × 10 completions).
  6. Write the JSON in the schema the aggregator expects:
     ``{persona_name: {"rate": float, "hits": int, "total": int}, ...}``
     plus reproducibility metadata.
  7. Persist raw completions to ``raw_completions.json`` so any future
     temperature-1 cross-check (the C4 confound) can reuse the same
     per-state outputs without re-running vLLM.
  8. Cleanup the local merged dir.

Per-state wall: ~5-7 min on 1× H100 (vLLM dominates).

Invocation::

    uv run python scripts/measure_leakage_228.py \\
        --source villain --checkpoint-step 200 --gpu-id 0 \\
        --output-dir eval_results/causal_proximity/strong_convergence
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _bootstrap import PROJECT_ROOT, bootstrap

bootstrap()

# Project imports must come AFTER bootstrap()
sys.path.insert(0, str(PROJECT_ROOT / "src"))
# Reuse #228 adapter map / constants
from compute_js_convergence_228 import (  # noqa: E402
    ADAPTER_MAP,
    ADAPTER_REPO,
    BASE_MODEL,
    CHECKPOINT_STEPS,
    HF_DTYPE,
    TMP_ROOT,
)

from explore_persona_space.personas import (  # noqa: E402
    ALL_EVAL_PERSONAS,
    EVAL_QUESTIONS,
    MARKER_TOKEN,
)

logger = logging.getLogger("measure_leakage_228")

# ── Constants (frozen to match #109's eval_causal_ckpt.py exactly) ────────

NUM_COMPLETIONS = 10
EVAL_TEMPERATURE = 1.0
EVAL_TOP_P = 1.0
MAX_NEW_TOKENS = 512
GEN_SEED = 42

VLLM_GPU_MEM_UTIL_DEFAULT = 0.85
VLLM_MAX_MODEL_LEN = 2048
VLLM_MAX_NUM_SEQS = 64

TMP_PHASE05_ROOT = TMP_ROOT.parent / "issue228_leakage"

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "causal_proximity" / "strong_convergence"


# ── Helpers ───────────────────────────────────────────────────────────────


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.warning("git rev-parse failed: %s", exc)
        return "unknown"


def _lib_versions() -> dict[str, str]:
    import peft
    import transformers
    import vllm

    return {
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "vllm": vllm.__version__,
        "torch": torch.__version__,
    }


# ── Adapter download + merge ──────────────────────────────────────────────


def _download_adapter(repo_id: str, subpath: str, local_root: Path) -> Path:
    """Download a single adapter subfolder from HF Hub. Returns local dir."""
    from huggingface_hub import snapshot_download

    local_root.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subpath}/*"],
        local_dir=str(local_root),
        token=os.environ.get("HF_TOKEN"),
    )
    adapter_local = local_root / subpath
    if not (adapter_local / "adapter_config.json").exists():
        raise RuntimeError(f"Adapter download missing adapter_config.json at {adapter_local}")
    return adapter_local


def _merge_two_adapters(
    convergence_subpath: str | None,
    marker_subpath: str,
    merged_dir: Path,
    gpu_id: int,
) -> None:
    """Merge convergence (optional) + marker adapters into base.

    For ``checkpoint_step > 0``: load base, apply convergence, merge,
    apply marker, merge, save.
    For ``checkpoint_step == 0``: load base, apply marker only, merge,
    save.
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    download_root = TMP_PHASE05_ROOT / "hf_dl" / merged_dir.name
    if download_root.exists():
        shutil.rmtree(download_root, ignore_errors=True)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=HF_DTYPE,
        device_map={"": f"cuda:{gpu_id}"},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )

    # Step A: convergence adapter (if any)
    if convergence_subpath is not None:
        conv_local = _download_adapter(ADAPTER_REPO, convergence_subpath, download_root)
        logger.info("Applying convergence adapter %s", conv_local)
        peft_model = PeftModel.from_pretrained(model, str(conv_local))
        model = peft_model.merge_and_unload()
        del peft_model
        gc.collect()
        torch.cuda.empty_cache()

    # Step B: marker adapter (always present)
    marker_local = _download_adapter(ADAPTER_REPO, marker_subpath, download_root)
    logger.info("Applying marker adapter %s", marker_local)
    peft_model = PeftModel.from_pretrained(model, str(marker_local))
    merged = peft_model.merge_and_unload()

    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(merged_dir))

    del peft_model, merged, model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    shutil.rmtree(download_root, ignore_errors=True)
    logger.info("Merged model saved to %s", merged_dir)


# ── vLLM generation (mirrors eval_causal_ckpt.py behaviour) ───────────────


def _generate_completions(
    model_path: str,
    gpu_mem_util: float,
    seed: int,
) -> dict[str, dict[str, list[str]]]:
    """vLLM-batched generation: 11 personas × 20 questions × 10 completions.

    Returns ``completions[persona_name][question] = [str, ...]``.
    """
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    prompt_texts: list[str] = []
    prompt_keys: list[tuple[str, str]] = []
    for persona_name, persona_prompt in ALL_EVAL_PERSONAS.items():
        for question in EVAL_QUESTIONS:
            messages = [
                {"role": "system", "content": persona_prompt},
                {"role": "user", "content": question},
            ]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(text)
            prompt_keys.append((persona_name, question))

    logger.info(
        "vLLM: %d prompts × %d completions = %d generations",
        len(prompt_texts),
        NUM_COMPLETIONS,
        len(prompt_texts) * NUM_COMPLETIONS,
    )
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        trust_remote_code=True,
        gpu_memory_utilization=gpu_mem_util,
        max_model_len=VLLM_MAX_MODEL_LEN,
        max_num_seqs=VLLM_MAX_NUM_SEQS,
        seed=seed,
    )
    sampling = SamplingParams(
        n=NUM_COMPLETIONS,
        temperature=EVAL_TEMPERATURE,
        top_p=EVAL_TOP_P,
        max_tokens=MAX_NEW_TOKENS,
        seed=seed,
    )
    try:
        outputs = llm.generate(prompt_texts, sampling)
    finally:
        del llm
        gc.collect()
        torch.cuda.empty_cache()

    results: dict[str, dict[str, list[str]]] = {name: {} for name in ALL_EVAL_PERSONAS}
    for output, (persona_name, question) in zip(outputs, prompt_keys, strict=True):
        results[persona_name][question] = [o.text for o in output.outputs]
    return results


# ── Marker rate computation ───────────────────────────────────────────────


def _marker_rates_per_persona(
    completions: dict[str, dict[str, list[str]]],
) -> dict[str, dict]:
    """Compute marker hit-rate per persona, aggregated over all questions.

    Returns the schema the aggregator (`aggregate_issue228._leakage_rate`)
    expects: ``{persona_name: {"rate": float, "hits": int, "total": int}}``.
    """
    out: dict[str, dict] = {}
    for persona_name, qmap in completions.items():
        hits = 0
        total = 0
        for _question, comps in qmap.items():
            for c in comps:
                total += 1
                if MARKER_TOKEN.lower() in c.lower():
                    hits += 1
        rate = hits / total if total > 0 else 0.0
        out[persona_name] = {
            "rate": float(rate),
            "hits": int(hits),
            "total": int(total),
        }
        logger.info("  %-22s rate=%.2f%% (%d/%d)", persona_name, rate * 100, hits, total)
    return out


# ── Top-level worker ──────────────────────────────────────────────────────


def _state_output_path(output_dir: Path, source: str, step: int) -> Path:
    return output_dir / source / f"checkpoint-{step}" / "marker_eval.json"


def measure_one_state(
    source: str,
    step: int,
    gpu_id: int,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    gpu_mem_util: float = VLLM_GPU_MEM_UTIL_DEFAULT,
    seed: int = GEN_SEED,
    skip_if_exists: bool = True,
) -> Path:
    """Measure leakage at 11 targets for one state. Idempotent.

    Returns the path to ``marker_eval.json``.
    """
    if source not in ADAPTER_MAP:
        raise ValueError(f"Unknown source {source!r}")
    if step != 0 and step not in CHECKPOINT_STEPS:
        raise ValueError(f"Unknown checkpoint_step={step}; expected 0 or one of {CHECKPOINT_STEPS}")

    out_path = _state_output_path(output_dir, source, step)
    if skip_if_exists and out_path.exists():
        logger.info("[%s ckpt-%d] marker_eval.json exists, skipping", source, step)
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if "CUDA_VISIBLE_DEVICES" in os.environ:
        visible = os.environ["CUDA_VISIBLE_DEVICES"].split(",")
        if len(visible) == 1:
            gpu_id = 0

    t_start = time.time()
    started_at = datetime.now(UTC).isoformat()

    # 1. Determine adapter subpaths.
    convergence_subpath: str | None
    if step == 0:
        convergence_subpath = None
    else:
        _, conv_subpath, _ = ADAPTER_MAP[source]
        convergence_subpath = f"{conv_subpath}/checkpoint-{step}"
    marker_subpath = f"adapters/cp_marker_{source}_ep{step}_s{seed}"

    # 2. Merge into a fresh dir. The whole pipeline below is wrapped in
    # ``try/finally`` so ``merged_dir`` is ALWAYS torn down — even on
    # exception (R6 disk-hygiene fix). Without this, a Phase-0.5 worker
    # that crashes mid-vLLM (OOM, KeyError, etc.) leaks ~14 GB of merged
    # safetensors per state and `/workspace` fills before the sweep
    # finishes.
    merged_dir = TMP_PHASE05_ROOT / f"{source}_ckpt{step}_full_merged"
    if merged_dir.exists():
        shutil.rmtree(merged_dir, ignore_errors=True)
    try:
        _merge_two_adapters(
            convergence_subpath=convergence_subpath,
            marker_subpath=marker_subpath,
            merged_dir=merged_dir,
            gpu_id=gpu_id,
        )

        # 3. vLLM generate.
        completions = _generate_completions(
            model_path=str(merged_dir),
            gpu_mem_util=gpu_mem_util,
            seed=seed,
        )

        # 4. Marker rates.
        rates = _marker_rates_per_persona(completions)

        # 5. Write the aggregator-compatible JSON (schema verified by tests).
        elapsed = time.time() - t_start
        completed_at = datetime.now(UTC).isoformat()
        payload: dict = {
            # Top-level: persona → {rate, hits, total}. The aggregator uses this
            # exact shape via ``_leakage_rate(leakage, persona) -> leakage[persona]['rate']``.
            **rates,
            # Reproducibility metadata under a reserved key the aggregator ignores.
            "_meta": {
                "source": source,
                "checkpoint_step": step,
                "convergence_adapter": convergence_subpath,
                "marker_adapter": marker_subpath,
                "base_model": BASE_MODEL,
                "n_completions_per_prompt": NUM_COMPLETIONS,
                "temperature": EVAL_TEMPERATURE,
                "top_p": EVAL_TOP_P,
                "max_tokens": MAX_NEW_TOKENS,
                "seed": seed,
                "n_personas": len(ALL_EVAL_PERSONAS),
                "n_questions": len(EVAL_QUESTIONS),
                "wall_seconds": elapsed,
                "git_commit": _git_commit(),
                "started_at": started_at,
                "completed_at": completed_at,
                **_lib_versions(),
            },
        }
        out_path.write_text(json.dumps(payload, indent=2))
        logger.info("[%s ckpt-%d] wrote %s (%.1fs)", source, step, out_path, elapsed)

        # 6. Persist raw completions for future temp=1 reanalysis (C4 confound).
        raw_path = out_path.parent / "raw_completions.json"
        raw_path.write_text(json.dumps(completions))
        logger.info("[%s ckpt-%d] wrote raw completions to %s", source, step, raw_path)

        return out_path
    finally:
        if merged_dir.exists():
            shutil.rmtree(merged_dir, ignore_errors=True)
            logger.info("[%s ckpt-%d] cleaned merged dir %s", source, step, merged_dir)
        gc.collect()
        torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help=f"One of {sorted(ADAPTER_MAP)}")
    parser.add_argument(
        "--checkpoint-step",
        type=int,
        required=True,
        help=f"0 (epoch-0 baseline) or one of {CHECKPOINT_STEPS}",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--gpu-mem-util", type=float, default=VLLM_GPU_MEM_UTIL_DEFAULT)
    parser.add_argument("--seed", type=int, default=GEN_SEED)
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Recompute even if marker_eval.json exists",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out_path = measure_one_state(
        source=args.source,
        step=args.checkpoint_step,
        gpu_id=args.gpu_id,
        output_dir=args.output_dir,
        gpu_mem_util=args.gpu_mem_util,
        seed=args.seed,
        skip_if_exists=not args.no_skip_existing,
    )
    print(f"OK {args.source} ckpt-{args.checkpoint_step} {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
