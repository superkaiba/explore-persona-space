#!/usr/bin/env python3
"""Issue #240 eval wrapper: quantized prefixes + GCG cells.

Evaluates both quantized soft-prefix strings (Part A) and GCG-optimized
discrete suffixes (Part B) through the standard 52-prompt Betley+Wang panel
with N=20, T=1.0, dual judge (Sonnet 4.5 + Opus 4.7), and classifier-C.

For quantized cells, two eval paths are run:
  - **vLLM path**: quantized string as a normal system_prompt via vLLM.
    This is the load-bearing measurement (real prompt channel).
  - **Token-ID path**: quantized token IDs via evaluate_gcg_with_token_ids.
    Enables within-backend comparison to #170's soft cells.

For GCG cells, the token-ID path is used (matching GCG training backend).

Usage:
    # Eval quantized cells (vLLM path):
    uv run python scripts/eval_issue240.py --mode quantized-vllm

    # Eval quantized cells (token-ID path):
    uv run python scripts/eval_issue240.py --mode quantized-tokenid

    # Eval GCG cells:
    uv run python scripts/eval_issue240.py --mode gcg

    # Eval everything:
    uv run python scripts/eval_issue240.py --mode all
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _bootstrap import bootstrap

bootstrap()

# hot-fix: transformers 5.5 removed all_special_tokens_extended; vLLM 0.11 needs it
from transformers import PreTrainedTokenizerBase  # noqa: E402

if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
    PreTrainedTokenizerBase.all_special_tokens_extended = PreTrainedTokenizerBase.all_special_tokens

import logging  # noqa: E402

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EVAL_DIR = Path("eval_results/issue-240")

JUDGE_MODELS = {
    "sonnet": "claude-sonnet-4-5-20250929",
    "opus": "claude-opus-4-7",
}

NUM_SAMPLES = 20
TEMPERATURE = 1.0
TOP_P = 0.95
MAX_NEW_TOKENS = 256

GCG_CELLS = ["hardL20_batched", "hardL40_batched", "hardL80_batched"]


def parse_args():
    p = argparse.ArgumentParser(description="Issue #240 eval wrapper")
    p.add_argument(
        "--mode",
        choices=["quantized-vllm", "quantized-tokenid", "gcg", "all"],
        default="all",
        help="Which eval to run.",
    )
    p.add_argument(
        "--quantized-json",
        default="eval_results/issue-240/quantized/quantized_strings.json",
        help="Path to quantized_strings.json from quantize_prefix.py.",
    )
    p.add_argument(
        "--gcg-dir",
        default="eval_results/issue-240/gcg",
        help="Directory containing GCG cell outputs.",
    )
    p.add_argument("--base-model", default=BASE_MODEL)
    p.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    p.add_argument("--temperature", type=float, default=TEMPERATURE)
    p.add_argument("--top-p", type=float, default=TOP_P)
    p.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument(
        "--judges",
        default="sonnet,opus",
        help="Comma-separated judge names (sonnet, opus).",
    )
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


# ── Eval helpers ─────────────────────────────────────────────────────────────


async def eval_quantized_vllm(
    *,
    args,
    cell: str,
    quantized_text: str,
    prompts: list[str],
    eval_name: str,
    judge_key: str,
) -> dict:
    """Evaluate a quantized prefix string via the standard vLLM path."""
    from explore_persona_space.eval.alignment import evaluate_alignment

    judge_model = JUDGE_MODELS[judge_key]
    out_dir = EVAL_DIR / "quantized" / cell / "vllm" / judge_key
    out_dir.mkdir(parents=True, exist_ok=True)

    return await evaluate_alignment(
        model_path=args.base_model,
        output_dir=str(out_dir),
        questions=prompts,
        eval_name=eval_name,
        judge_model=judge_model,
        num_samples=args.num_samples,
        temperature=args.temperature,
        seed=42,
        system_prompt=quantized_text,
    )


async def eval_quantized_tokenid(
    *,
    args,
    cell: str,
    token_ids: list[int],
    prompts: list[str],
    eval_name: str,
    judge_key: str,
) -> dict:
    """Evaluate quantized token IDs via token-ID pass-through (matches #170 backend)."""
    from explore_persona_space.eval.alignment import evaluate_gcg_with_token_ids

    judge_model = JUDGE_MODELS[judge_key]
    out_dir = EVAL_DIR / "quantized" / cell / "tokenid" / judge_key
    out_dir.mkdir(parents=True, exist_ok=True)

    return await evaluate_gcg_with_token_ids(
        base_model_path=args.base_model,
        suffix_ids=token_ids,
        output_dir=str(out_dir),
        questions=prompts,
        eval_name=eval_name,
        judge_model=judge_model,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=42,
        slot="system",
    )


async def eval_gcg_cell(
    *,
    args,
    cell: str,
    suffix_ids: list[int],
    prompts: list[str],
    eval_name: str,
    judge_key: str,
) -> dict:
    """Evaluate a GCG-optimized suffix via token-ID pass-through."""
    from explore_persona_space.eval.alignment import evaluate_gcg_with_token_ids

    judge_model = JUDGE_MODELS[judge_key]
    out_dir = EVAL_DIR / "gcg" / cell / judge_key
    out_dir.mkdir(parents=True, exist_ok=True)

    return await evaluate_gcg_with_token_ids(
        base_model_path=args.base_model,
        suffix_ids=suffix_ids,
        output_dir=str(out_dir),
        questions=prompts,
        eval_name=eval_name,
        judge_model=judge_model,
        num_samples=args.num_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        seed=42,
        slot="system",
    )


# ── Dual-panel eval helper ───────────────────────────────────────────────────


async def _eval_dual_panel(
    eval_fn,
    label: str,
    cell: str,
    judge: str,
    betley_prompts: list[str],
    wang_prompts: list[str],
    **kwargs,
) -> dict:
    """Run Betley + Wang dual-panel eval for one cell/judge, return scores dict."""
    logger.info("Eval %s: cell=%s judge=%s", label, cell, judge)
    print(f"\n=== {label} | {cell} | {judge} | Betley ===", flush=True)
    b_summary = await eval_fn(
        cell=cell, eval_name="betley", judge_key=judge, prompts=betley_prompts, **kwargs
    )
    print(f"  Betley aligned={b_summary['overall_mean_aligned']:.2f}")

    print(f"=== {label} | {cell} | {judge} | Wang ===", flush=True)
    w_summary = await eval_fn(
        cell=cell, eval_name="wang", judge_key=judge, prompts=wang_prompts, **kwargs
    )
    print(f"  Wang aligned={w_summary['overall_mean_aligned']:.2f}")

    return {
        "betley": b_summary["overall_mean_aligned"],
        "wang": w_summary["overall_mean_aligned"],
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def _resolve_local_model(model_name: str) -> str:
    """Resolve HF model name to local snapshot path if cached (avoids vLLM tqdm bug)."""
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    cache_dir = os.path.join(
        hf_home, "hub", f"models--{model_name.replace('/', '--')}", "snapshots"
    )
    if os.path.isdir(cache_dir):
        snaps = os.listdir(cache_dir)
        if snaps:
            resolved = os.path.join(cache_dir, snaps[0])
            logger.info("Resolved %s -> %s", model_name, resolved)
            return resolved
    return model_name


async def main_async():  # noqa: C901
    args = parse_args()
    # hot-fix: resolve model to local path to avoid vLLM tqdm download bug
    args.base_model = _resolve_local_model(args.base_model)
    judges = [j.strip() for j in args.judges.split(",")]
    for j in judges:
        if j not in JUDGE_MODELS:
            raise ValueError(f"Unknown judge {j!r}; available: {list(JUDGE_MODELS)}")

    # Load prompts.
    from explore_persona_space.eval.alignment import WANG_PROMPTS
    from explore_persona_space.personas import BETLEY_QUESTIONS as BETLEY_MAIN_QUESTIONS

    betley_prompts = list(BETLEY_MAIN_QUESTIONS)
    wang_prompts = list(WANG_PROMPTS)

    modes = [args.mode] if args.mode != "all" else ["quantized-vllm", "quantized-tokenid", "gcg"]
    rollup: dict[str, dict] = {}
    t0 = time.time()

    # ── Quantized cell evals ─────────────────────────────────────────────
    if "quantized-vllm" in modes or "quantized-tokenid" in modes:
        qpath = Path(args.quantized_json)
        if not qpath.exists():
            raise FileNotFoundError(
                f"Quantized strings JSON not found at {qpath}. "
                "Run scripts/quantize_prefix.py first."
            )
        quantized = json.loads(qpath.read_text())

        for cell, qdata in quantized.items():
            for judge in judges:
                if "quantized-vllm" in modes:
                    scores = await _eval_dual_panel(
                        eval_quantized_vllm,
                        "Quantized vLLM",
                        cell,
                        judge,
                        betley_prompts,
                        wang_prompts,
                        args=args,
                        quantized_text=qdata["decoded_string"],
                    )
                    rollup.setdefault(f"quantized_vllm_{cell}", {})[judge] = scores

                if "quantized-tokenid" in modes:
                    scores = await _eval_dual_panel(
                        eval_quantized_tokenid,
                        "Quantized Token-ID",
                        cell,
                        judge,
                        betley_prompts,
                        wang_prompts,
                        args=args,
                        token_ids=qdata["token_ids"],
                    )
                    rollup.setdefault(f"quantized_tokenid_{cell}", {})[judge] = scores

    # ── GCG cell evals ───────────────────────────────────────────────────
    if "gcg" in modes:
        gcg_dir = Path(args.gcg_dir)
        for cell in GCG_CELLS:
            suffix_path = gcg_dir / cell / "best_suffix.json"
            if not suffix_path.exists():
                logger.warning(
                    "GCG cell %s: best_suffix.json not found at %s, skipping",
                    cell,
                    suffix_path,
                )
                continue
            suffix_obj = json.loads(suffix_path.read_text())
            suffix_ids = suffix_obj.get("suffix_ids")
            if suffix_ids is None:
                raise RuntimeError(f"{suffix_path} has no 'suffix_ids' key")

            for judge in judges:
                scores = await _eval_dual_panel(
                    eval_gcg_cell,
                    "GCG",
                    cell,
                    judge,
                    betley_prompts,
                    wang_prompts,
                    args=args,
                    suffix_ids=suffix_ids,
                )
                rollup.setdefault(f"gcg_{cell}", {})[judge] = scores

    # ── Classifier-C scoring ────────────────────────────────────────────
    classifier_path = Path(
        os.environ.get(
            "CLASSIFIER_C_PATH",
            "/workspace/explore-persona-space/eval_results/issue-104/classifier_c.joblib",
        )
    )
    if classifier_path.exists():
        logger.info("PHASE: Classifier-C scoring via %s", classifier_path)
        from joblib import load as joblib_load

        bundle = joblib_load(str(classifier_path))
        clf = bundle["classifier"]
        encoder_name = bundle["encoder_name"]
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer(encoder_name)

        # Score completions stored during eval (read from per-cell JSONs)
        completions_dir = EVAL_DIR
        for key in list(rollup.keys()):
            comp_path = completions_dir / key / "completions.json"
            if comp_path.exists():
                import numpy as np

                comps = json.loads(comp_path.read_text())
                all_texts = [c for texts in comps.values() for c in texts]
                if all_texts:
                    embeds = encoder.encode(all_texts, show_progress_bar=False)
                    probs = clf.predict_proba(embeds)[:, 1]
                    rollup[key]["classifier_c"] = {
                        "mean_c": float(np.mean(probs)),
                        "std_c": float(np.std(probs)),
                        "n": len(probs),
                    }
                    logger.info("  %s: C=%.4f (n=%d)", key, np.mean(probs), len(probs))
            else:
                logger.warning("  %s: no completions.json, skipping classifier-C", key)
    else:
        logger.warning("Classifier-C not found at %s, skipping", classifier_path)

    # ── Save rollup ──────────────────────────────────────────────────────
    rollup_path = EVAL_DIR / "eval_rollup.json"
    rollup_path.parent.mkdir(parents=True, exist_ok=True)
    rollup_path.write_text(json.dumps(rollup, indent=2))
    dt = time.time() - t0
    logger.info("Eval complete in %.1fs. Rollup at %s", dt, rollup_path)

    # Print summary.
    print(f"\n=== Eval Rollup ({dt:.0f}s) ===")
    for key, judges_dict in rollup.items():
        for judge, scores in judges_dict.items():
            alpha = 100 * (1 - (scores["betley"] * 8 + scores["wang"] * 44) / 52)
            print(f"  {key} | {judge}: alpha={alpha:.1f}")


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set in env")
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
