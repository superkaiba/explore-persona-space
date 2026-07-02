#!/usr/bin/env python
"""Issue #778 Phase 2 — system-prompt / monitoring prediction, per trait.

The paper's monitoring setup (§"Monitoring prompt-induced persona shifts"):
  - N system prompts (the released 5 neg + 5 pos induction instructions; see
    issue778_lib.monitoring_system_prompts for the 10-vs-paper's-8 deviation)
    x 20 evaluation-set questions -> cells.
  - 10 on-policy rollouts per cell (vLLM), graded 0-100 judge score (Sonnet 4.5,
    N=6 draws @ T=0.7, mean-aggregated) -> the trait-expression DV per cell.
  - Projection = last-prompt-token activation onto r_B[layer], at every layer.

Output (per trait): ``eval_results/issue_778/monitoring_{trait}.jsonl`` — one line
per cell carrying (prompt_id, question, mean_trait_score, per_rollout_scores,
projection_per_layer, condition_id). The null battery consumes these + the cached
r_B / activation pools.

r_B must already exist under ``data/issue_778/rb/{trait}.pt`` (Phase 1).
"""

from __future__ import annotations

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib

from explore_persona_space.orchestrate.env import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue778.monitoring")
load_dotenv()


def _chat_prompt(tokenizer, system: str, question: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _vllm_generate(llm, prompts: list[str], *, temperature: float, max_new: int) -> list[str]:
    from vllm import SamplingParams

    chunk_size = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_new, min_tokens=1)
    out: list[str] = []
    n_chunks = (len(prompts) + chunk_size - 1) // chunk_size
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i : i + chunk_size]
        logger.info(
            "[vllm-chunk] monitoring chunk %d/%d (%d)", i // chunk_size + 1, n_chunks, len(chunk)
        )
        res = llm.generate(chunk, sp, use_tqdm=False)
        out.extend(o.outputs[0].text for o in res)
    return out


def monitor_trait(
    trait: str,
    external_root: Path,
    out_root: Path,
    eval_results_root: Path,
    *,
    n_questions: int,
    n_prompts: int,
    n_rollouts: int,
) -> dict:
    """Run the monitoring setup for one trait; write per-cell JSONL."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lib.log_phase("monitoring", f"trait={trait} start", trait=trait)
    td = lib.load_trait_data(external_root, trait)
    tokenizer = AutoTokenizer.from_pretrained(lib.MODEL_NAME)

    system_prompts = lib.monitoring_system_prompts(td)[:n_prompts]
    questions = td.eval_questions[:n_questions]

    # Build cells: (prompt_id, system, question). Each cell -> n_rollouts.
    cells: list[dict] = []
    for prompt_id, system in system_prompts:
        for q in questions:
            cells.append({"prompt_id": prompt_id, "system": system, "question": q})

    # ── vLLM generation: n_rollouts per cell ─────────────────────────────────
    llm = lib.build_vllm_engine()
    try:
        gen_prompts: list[str] = []
        for cell in cells:
            chat = _chat_prompt(tokenizer, cell["system"], cell["question"])
            cell["prompt_text"] = chat
            gen_prompts.extend([chat] * n_rollouts)
        answers = _vllm_generate(
            llm, gen_prompts, temperature=lib.EXTRACT_TEMPERATURE, max_new=lib.MAX_NEW_TOKENS
        )
    finally:
        lib.reap_vllm_engine(llm)
    # Slice answers back per cell.
    for ci, cell in enumerate(cells):
        cell["answers"] = answers[ci * n_rollouts : (ci + 1) * n_rollouts]
    lib.log_phase("monitoring", f"trait={trait} generated {len(answers)}", trait=trait)

    # ── Graded judge (N=6 draws per rollout, mean over rollouts+draws) ───────
    cache_dir = out_root / "monitoring" / f"{trait}_judge_cache"
    save_raw = out_root / "monitoring" / f"{trait}_judge_raw.json"
    save_raw.parent.mkdir(parents=True, exist_ok=True)
    judge_items: list[tuple[str, str, str]] = []
    for ci, cell in enumerate(cells):
        for ri, ans in enumerate(cell["answers"]):
            item_id = f"c{ci:04d}-r{ri:02d}"
            judge_items.append((item_id, cell["question"], ans))
    jr = lib.judge_graded(
        judge_items,
        td.eval_prompt,
        n_draws=lib.JUDGE_N_DRAWS,
        cache_dir=cache_dir,
        save_raw=save_raw,
        temperature=lib.JUDGE_TEMPERATURE,
    )
    # Per cell: mean over the kept per-rollout scores (each already mean over draws).
    for ci, cell in enumerate(cells):
        per_rollout: list[float | None] = []
        for ri in range(len(cell["answers"])):
            per_rollout.append(jr.scores.get(f"c{ci:04d}-r{ri:02d}"))
        kept = [s for s in per_rollout if s is not None]
        cell["per_rollout_scores"] = per_rollout
        cell["mean_trait_score"] = (sum(kept) / len(kept)) if kept else None
    n_null_cells = sum(1 for c in cells if c["mean_trait_score"] is None)
    if n_null_cells == len(cells):
        raise RuntimeError(f"trait={trait}: every monitoring cell dropped by judge — check wiring")

    # ── Last-prompt-token activation capture (all layers), per cell ──────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        lib.MODEL_NAME, torch_dtype=dtype, device_map=device if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to(device)
    try:
        prompts = [cell["prompt_text"] for cell in cells]
        acts = lib.capture_last_prompt_token_all_layers(
            model, tokenizer, prompts, device=model.device
        )  # (n_cells, 28, 3584)
    finally:
        del model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # Persist the RAW last-prompt activation tensor (n_cells, 28, 3584), aligned
    # with the JSONL row order (pre-drop). The null battery re-projects these raw
    # activations onto each null direction — the stored projections are onto r_B
    # only. Plan-referenced downstream input -> uploads to analysis_tensors/.
    (out_root / "monitoring").mkdir(parents=True, exist_ok=True)
    torch.save(acts, out_root / "monitoring" / f"{trait}_acts.pt")

    # ── Projection per layer onto r_B ────────────────────────────────────────
    rb = torch.load(out_root / "rb" / f"{trait}.pt", weights_only=False)  # (28, 3584)
    import numpy as np

    acts_np = acts.numpy().astype(np.float64)  # (n_cells, 28, 3584)
    rb_np = rb.numpy().astype(np.float64)
    from explore_persona_space.analysis.null_battery import project

    # projection[cell, layer] = a_proj_b(act[cell,layer], rb[layer])
    proj = np.empty((len(cells), lib.N_LAYERS), dtype=np.float64)
    for layer in range(lib.N_LAYERS):
        proj[:, layer] = project(acts_np[:, layer, :], rb_np[layer])

    # ── Write per-cell JSONL ─────────────────────────────────────────────────
    eval_results_root.mkdir(parents=True, exist_ok=True)
    out_path = eval_results_root / f"monitoring_{trait}.jsonl"
    prompt_id_to_int = {pid: i for i, (pid, _s) in enumerate(system_prompts)}
    with open(out_path, "w") as f:
        for ci, cell in enumerate(cells):
            row = {
                "trait": trait,
                "prompt_id": cell["prompt_id"],
                "condition_id": prompt_id_to_int[cell["prompt_id"]],
                "question": cell["question"],
                "mean_trait_score": cell["mean_trait_score"],
                "per_rollout_scores": cell["per_rollout_scores"],
                "projection_per_layer": proj[ci].tolist(),
            }
            f.write(json.dumps(row) + "\n")

    meta = {
        "trait": trait,
        "n_cells": len(cells),
        "n_prompts": len(system_prompts),
        "n_questions": len(questions),
        "n_rollouts_per_cell": n_rollouts,
        "n_cells_dropped_all_judge": n_null_cells,
        "judge_draws_total": jr.n_total_draws,
        "judge_draws_dropped": jr.n_dropped_draws,
        "out_path": str(out_path),
        "reproducibility": lib.repro_metadata(),
    }
    with open(out_root / "monitoring" / f"{trait}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    lib.log_phase("monitoring", f"trait={trait} done", trait=trait, n_cells=len(cells))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #778 Phase 2 monitoring.")
    parser.add_argument("--external-root", default="external/persona_vectors")
    parser.add_argument("--out-root", default="data/issue_778")
    parser.add_argument("--eval-results-root", default="eval_results/issue_778")
    parser.add_argument("--traits", nargs="+", default=list(lib.TRAITS))
    parser.add_argument("--cells", type=int, default=None, help="limit to first N traits (smoke)")
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--n-prompts", type=int, default=10, help="system prompts (5 neg + 5 pos)")
    parser.add_argument("--n-rollouts", type=int, default=lib.N_ROLLOUTS_PRED)
    args = parser.parse_args()

    external_root = Path(args.external_root)
    out_root = Path(args.out_root)
    eval_results_root = Path(args.eval_results_root)
    traits = args.traits
    if args.cells is not None:
        traits = traits[: args.cells]

    lib.log_phase("monitoring", f"start traits={traits}")
    results = {}
    for trait in traits:
        results[trait] = monitor_trait(
            trait,
            external_root,
            out_root,
            eval_results_root,
            n_questions=args.n_questions,
            n_prompts=args.n_prompts,
            n_rollouts=args.n_rollouts,
        )
    lib.log_phase("monitoring", f"all traits done ({len(results)})")
    print(json.dumps({"phase": "monitoring", "traits": list(results)}, indent=2))


if __name__ == "__main__":
    main()
