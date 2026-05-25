#!/usr/bin/env python3
"""Per-checkpoint marker-emission eval driver for issue #385.

For each LoRA checkpoint in --steps, generates n=8 completions for each
(bystander x prompt) cell across the 27-row bystander panel
(19 personas + 8 non-persona contexts), then scores marker
([ZLT]) emission rate per cell.

Total per checkpoint = 27 bystanders x 20 prompts x 8 completions = 4,320
completions. Across 14 default checkpoints = 60,480 completions.

vLLM batches all (bystander, prompt) prompts within a single checkpoint's
LLM.generate() call. The LoRA adapter is swapped per checkpoint via
LoRARequest(lora_path=<checkpoint_dir>) without rebuilding the engine.

CRITICAL (CLAUDE.md "checkpoint per phase", incident #377): each
checkpoint's per-bystander result + raw completions is written to disk AS
SOON AS THE CHECKPOINT FINISHES, so a crash on checkpoint N+1 doesn't lose
results from checkpoints 0..N. Layout per run:

  <output_root>/issue_385_marker_spread/seed{S}/
    summary.json                                        # aggregated, written at end
    step{N}/
      marker_rates.json                                 # per-bystander rate (small)
      raw_completions.json                              # full text, ~10-15 MB
    step{N+1}/
      ...

Usage:
    uv run python scripts/eval_i385_marker_spread.py \\
      --run-dir <RUN_DIR_FROM_TRAINING> \\
      --steps 5,10,25,50,75,100,150,200,300,400,600,800,1200,1600 \\
      --output-root eval_results/issue_385 \\
      --seed 42
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SOURCE_PERSONA = "librarian"
MARKER = "[ZLT]"

# Eval sampling config (plan §5.3): T=1.0, top_p=1.0, max_tokens=512, n=8.
SAMPLING_TEMPERATURE = 1.0
SAMPLING_TOP_P = 1.0
SAMPLING_MAX_TOKENS = 512
SAMPLING_N = 8

# LoRA-engine config (plan §5.3): max_lora_rank=32 to match the LoRA r=32 used
# in training; one adapter loaded per LoRARequest call.
LORA_MAX_RANK = 32
GPU_MEMORY_UTILIZATION = 0.85


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _build_bystander_panel() -> tuple[list[tuple[str, str]], list[str]]:
    """Build the 27-row bystander panel (19 personas non-source + 8 contexts) + PROMPTS.

    Reuses scripts/compute_i385_predictors.py:_build_panel() and drops the
    source persona to leave exactly the bystander rows. The returned panel
    order matches the cosine/JS predictor file row order with the source
    persona removed.
    """
    # Importing the helper lazily so this script's --help works without torch.
    from compute_i385_predictors import _build_panel

    full_panel, prompts = _build_panel()
    bystanders = [(name, text) for name, text in full_panel if name != SOURCE_PERSONA]
    if len(bystanders) != 27:
        raise RuntimeError(f"Expected 27 bystander rows, got {len(bystanders)}")
    return bystanders, prompts


def _render_prompts(
    tokenizer, bystanders: list[tuple[str, str]], prompts: list[str]
) -> tuple[list[str], list[tuple[str, int]]]:
    """Render (bystander_system, user_prompt) chat templates.

    Returns (rendered_strings, keys) where keys[i] = (bystander_name, prompt_idx).
    """
    rendered: list[str] = []
    keys: list[tuple[str, int]] = []
    for bys_name, bys_system in bystanders:
        for p_idx, probe in enumerate(prompts):
            messages = []
            if bys_system:
                messages.append({"role": "system", "content": bys_system})
            messages.append({"role": "user", "content": probe})
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            rendered.append(text)
            keys.append((bys_name, p_idx))
    return rendered, keys


def _score_marker(
    completions_by_bystander: dict[str, dict[str, list[str]]],
    marker: str = MARKER,
) -> dict[str, dict]:
    """Wrapper around explore_persona_space.eval.trait_scorers.evaluate_markers.

    Importing inside this helper keeps the script's --help fast.
    """
    from explore_persona_space.eval.trait_scorers import evaluate_markers

    return evaluate_markers(completions_by_bystander, marker=marker)


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s (%.2f MB)", path, path.stat().st_size / 1e6)


def run_eval(args: argparse.Namespace) -> None:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"--run-dir does not exist: {run_dir}")
    steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    if not steps:
        raise SystemExit("--steps must be a non-empty comma-separated list")

    output_root = Path(args.output_root)
    run_output_dir = output_root / f"seed{args.seed}"
    run_output_dir.mkdir(parents=True, exist_ok=True)

    # ── Panel construction ────────────────────────────────────────────────────
    # _build_panel lives in scripts/compute_i385_predictors.py; add scripts/ to
    # sys.path so we can import it (sibling-script imports without a package).
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    try:
        bystanders, prompts = _build_bystander_panel()
    finally:
        sys.path.pop(0)

    logger.info(
        "Eval panel: %d bystanders x %d prompts x n=%d completions per cell",
        len(bystanders),
        len(prompts),
        SAMPLING_N,
    )
    logger.info("Checkpoints to evaluate: %s", steps)

    # Pre-flight: confirm all checkpoint directories exist BEFORE engine load
    # (avoids a 60s vLLM cold-start tax if one is missing).
    for step in steps:
        ckpt_path = run_dir / f"checkpoint-{step}"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Missing checkpoint dir: {ckpt_path}")
        adapter_safetensors = ckpt_path / "adapter_model.safetensors"
        if not adapter_safetensors.exists():
            raise FileNotFoundError(
                f"Checkpoint {ckpt_path} missing adapter_model.safetensors; "
                f"the SaveAtSpecificSteps callback may not have fired."
            )
    logger.info("All %d checkpoint adapter dirs present.", len(steps))

    # ── Engine load (once) ─────────────────────────────────────────────────────
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    logger.info(
        "Loading vLLM engine for %s (enable_lora=True, max_lora_rank=%d)...",
        BASE_MODEL,
        LORA_MAX_RANK,
    )
    t0 = time.time()
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=LORA_MAX_RANK,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        dtype="bfloat16",
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()
    logger.info("vLLM engine loaded in %.1fs", time.time() - t0)

    sampling = SamplingParams(
        n=SAMPLING_N,
        temperature=SAMPLING_TEMPERATURE,
        top_p=SAMPLING_TOP_P,
        max_tokens=SAMPLING_MAX_TOKENS,
        seed=args.seed,
    )

    rendered, keys = _render_prompts(tokenizer, bystanders, prompts)
    logger.info("Rendered %d prompts (1 per (bystander, prompt))", len(rendered))

    # ── Per-checkpoint loop ───────────────────────────────────────────────────
    # Persist each checkpoint's rates + raw completions BEFORE moving on.
    summary_rows: list[dict] = []
    for step in steps:
        adapter_path = run_dir / f"checkpoint-{step}"
        lora_req = LoRARequest(
            lora_name=f"i385_step{step}",
            lora_int_id=step,
            lora_path=str(adapter_path),
        )

        logger.info(
            "Step %d: launching vLLM.generate() on %d prompts x n=%d completions = %d total...",
            step,
            len(rendered),
            SAMPLING_N,
            len(rendered) * SAMPLING_N,
        )
        t_step = time.time()
        outputs = llm.generate(rendered, sampling, lora_request=lora_req)
        gen_secs = time.time() - t_step
        n_completions = sum(len(o.outputs) for o in outputs)
        logger.info(
            "Step %d generation done in %.1fs (%d completions, %.1f/s)",
            step,
            gen_secs,
            n_completions,
            n_completions / max(gen_secs, 1e-6),
        )

        # Collate per (bystander, prompt) -> n completions
        by_bys: dict[str, dict[str, list[str]]] = {
            n: {p: [] for p in prompts} for n, _ in bystanders
        }
        for out, (bys_name, p_idx) in zip(outputs, keys, strict=True):
            probe_text = prompts[p_idx]
            by_bys[bys_name][probe_text] = [o.text for o in out.outputs]

        # Score
        marker_results = _score_marker(by_bys, marker=MARKER)

        # Write per-step files IMMEDIATELY ── crash safety.
        step_dir = run_output_dir / f"step{step}"
        step_dir.mkdir(parents=True, exist_ok=True)

        # raw_completions.json: every completion string (for HF data-repo upload
        # via upload_raw_completions_to_data_repo, plan §11 Artifacts).
        raw_payload = {
            "metadata": {
                "step": step,
                "adapter_path": str(adapter_path),
                "base_model": BASE_MODEL,
                "marker": MARKER,
                "sampling": {
                    "n": SAMPLING_N,
                    "temperature": SAMPLING_TEMPERATURE,
                    "top_p": SAMPLING_TOP_P,
                    "max_tokens": SAMPLING_MAX_TOKENS,
                    "seed": args.seed,
                },
                "git_commit": _git_commit(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            "completions": by_bys,
        }
        _write_json(raw_payload, step_dir / "raw_completions.json")

        # marker_rates.json: per-bystander rates + per-question breakdown
        # (small; safe to read in analysis script without loading raw text).
        rates_payload = {
            "metadata": {
                "step": step,
                "adapter_path": str(adapter_path),
                "marker": MARKER,
                "n_per_cell": SAMPLING_N * len(prompts),
                "n_prompts": len(prompts),
                "n_bystanders": len(bystanders),
                "git_commit": _git_commit(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            "per_bystander": marker_results,
        }
        _write_json(rates_payload, step_dir / "marker_rates.json")

        # Lightweight summary row (no raw text) accumulated for the aggregated file.
        summary_rows.append(
            {
                "step": step,
                "adapter_path": str(adapter_path),
                "gen_secs": gen_secs,
                "per_bystander_rate": {
                    name: float(marker_results[name]["rate"]) for name, _ in bystanders
                },
            }
        )
        # Persist the running summary at every step too (the rolling summary
        # protects against losing the aggregated view if the last step crashes).
        running_summary = {
            "metadata": {
                "run_dir": str(run_dir),
                "seed": args.seed,
                "steps_completed": [r["step"] for r in summary_rows],
                "steps_total": steps,
                "git_commit": _git_commit(),
                "timestamp_utc": datetime.now(UTC).isoformat(),
            },
            "bystanders": [n for n, _ in bystanders],
            "prompts": prompts,
            "rows": summary_rows,
        }
        _write_json(running_summary, run_output_dir / "summary.json")

        logger.info(
            "Step %d wall: %.1fs; cum %d/%d steps done.",
            step,
            time.time() - t_step,
            len(summary_rows),
            len(steps),
        )

    logger.info(
        "All %d checkpoints done. Summary at %s", len(steps), run_output_dir / "summary.json"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        required=True,
        help="Training run directory containing checkpoint-{step}/ adapter dirs.",
    )
    parser.add_argument(
        "--steps",
        required=True,
        help="Comma-separated list of step values to evaluate (e.g. '5,10,25,...,1600').",
    )
    parser.add_argument(
        "--output-root",
        default="eval_results/issue_385",
        help="Output root (default: eval_results/issue_385).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Eval sampling seed (forwarded to SamplingParams). Default 42.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    run_eval(args)


if __name__ == "__main__":
    # Silence harmless tokenizers-parallelism warning on multi-thread vLLM.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
