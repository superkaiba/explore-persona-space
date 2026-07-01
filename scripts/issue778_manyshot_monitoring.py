#!/usr/bin/env python
"""Issue #778 Leg B — many-shot ICL monitoring (corrected-monitoring-8prompt-ladder).

Extends #778's monitoring predictor to the paper's MANY-SHOT ICL setting (App
"Monitoring persona shifts"): instead of a trait-inducing SYSTEM prompt, the trait
is induced by prepending ``shot_count`` in-context exemplars (each an extraction
question + a kept-positive trait-exhibiting response) as prior chat turns, then the
model answers a held-out eval question on-policy. The predictor is the SAME
last-prompt-token activation . r_B[layer]; the DV is the SAME graded 0-100
Sonnet-4.5 trait score over R=10 on-policy rollouts x N=6 judge draws.

Two sub-phases per trait:
  1. Exemplar-pool REGEN (``regenerate_kept_positive_pool``): #778 persisted only
     activation tensors + per-rollout scores, NOT the extraction rollout RESPONSE
     TEXT, so the "cached kept-positive rollouts" the scope cites do not exist
     (plan v4 §12 Assumption 5). We regenerate them on-policy under the released 5
     positive extraction instructions x 20 EXTRACTION questions (disjoint from the
     20 EVAL questions), R=10 rollouts @ T=1.0, judge-filter keep score>50 (N=1,
     drop REFUSAL/OOR). Persisted ->
     ``data/issue_778/exemplar_pool/{trait}_kept_pos.json`` (the path
     ``issue778_upload_corrected.py`` uploads to ``extraction_rollouts_regen/``).
     Kill criterion K1: a pool < max-shot-count (20) after regen -> that trait's
     Leg B is reported N/A (never a fake zero), the other traits proceed.
  2. ICL monitoring: shot-count in {0,5,10,15,20}, sample exemplars WITHOUT
     replacement (seed 42 per (trait, shot_count)), truncate each exemplar response
     to ``--exemplar-max-tokens`` (default 300) so the 20-shot prompt fits the
     Leg-B vLLM ``max_model_len`` (default 8192). 0-shot = the eval question alone
     (plain-assistant baseline anchor, no system prompt, no exemplars).

Output (per trait): ``eval_results/issue_778/monitoring_manyshot_{trait}.jsonl`` —
one row per cell (condition_id = shot_count, shot_count, question, mean_trait_score,
per_rollout_scores, projection_per_layer) + the raw last-prompt activation tensor
``data/issue_778/monitoring_manyshot/{trait}_acts.pt`` (n_cells, 28, 3584). The null
battery consumes these via ``issue778_null_battery.py --input-tag monitoring_manyshot``.

r_B must already exist under ``data/issue_778/rb/{trait}.pt`` (reused from #778).
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
logger = logging.getLogger("issue778.manyshot")
load_dotenv()

OUT_TAG = "monitoring_manyshot"
DEFAULT_SHOT_COUNTS = (0, 5, 10, 15, 20)
DEFAULT_EXEMPLAR_MAX_TOKENS = 300  # trait expression is visible in the opening
DEFAULT_MAX_MODEL_LEN = 8192  # Leg-B ICL contexts overflow the #778 4096 window
EXEMPLAR_SEED = 42  # fixed per (trait, shot_count) sampling seed


def _vllm_generate(llm, prompts: list[str], *, temperature: float, max_new: int) -> list[str]:
    """Batched vLLM generation (chunked for the #664 large-batch-deadlock guard)."""
    from vllm import SamplingParams

    chunk_size = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_new, min_tokens=1)
    out: list[str] = []
    n_chunks = (len(prompts) + chunk_size - 1) // chunk_size
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i : i + chunk_size]
        logger.info(
            "[vllm-chunk] manyshot chunk %d/%d (%d)", i // chunk_size + 1, n_chunks, len(chunk)
        )
        res = llm.generate(chunk, sp, use_tqdm=False)
        out.extend(o.outputs[0].text for o in res)
    return out


def _reduce_cell_scores(cells: list[dict], judge_result) -> int:
    """Reduce per-rollout judge scores into per-cell mean; returns fully-dropped count.

    Mirrors the committed ``issue778_monitoring.py`` reduction (item ids
    ``f"c{ci:04d}-r{ri:02d}"``): sets ``per_rollout_scores`` (per-rollout, None if
    dropped) + ``mean_trait_score`` (mean over kept rollouts, None if none kept).
    """
    n_null = 0
    for ci, cell in enumerate(cells):
        per_rollout: list[float | None] = []
        for ri in range(len(cell["answers"])):
            per_rollout.append(judge_result.scores.get(f"c{ci:04d}-r{ri:02d}"))
        kept = [s for s in per_rollout if s is not None]
        cell["per_rollout_scores"] = per_rollout
        cell["mean_trait_score"] = (sum(kept) / len(kept)) if kept else None
        if cell["mean_trait_score"] is None:
            n_null += 1
    return n_null


def _truncate_response(tokenizer, text: str, max_tokens: int) -> str:
    """Truncate an exemplar response to the first ``max_tokens`` tokens.

    Keeps the opening (where the trait signature is most visible) and re-decodes
    so the ICL context stays a valid string. ``add_special_tokens=False`` so no
    BOS/EOS creeps in.
    """
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)


def regenerate_kept_positive_pool(
    trait: str,
    external_root: Path,
    out_root: Path,
    *,
    n_extract_questions: int,
    n_rollouts: int,
    llm=None,
) -> list[dict]:
    """Regenerate the on-policy kept-positive exemplar pool for one trait.

    For each of the 5 released POSITIVE extraction instructions x the first
    ``n_extract_questions`` EXTRACTION questions, generate ``n_rollouts`` on-policy
    rollouts under the positive extraction system prompt (T=1.0), judge-filter
    keep score>50 (N=1, drop REFUSAL/OOR). Each surviving rollout is one exemplar
    ``{"extract_question", "response", "instruction_idx", "score"}``.

    Persists the pool -> ``data/issue_778/exemplar_pool/{trait}_kept_pos.json`` and
    returns it. ``llm`` (an already-built vLLM engine) is reused if given; else one
    is built + reaped here.
    """
    from transformers import AutoTokenizer

    td = lib.load_trait_data(external_root, trait)
    extract_qs = td.extract_questions[:n_extract_questions]

    tokenizer = AutoTokenizer.from_pretrained(lib.MODEL_NAME)
    cells: list[dict] = []
    for instr_idx, instr in enumerate(td.pos_instructions):
        system = lib.extraction_system_prompt(trait, instr, "pos")
        for q in extract_qs:
            cells.append({"instr_idx": instr_idx, "system": system, "question": q})

    own_engine = llm is None
    if own_engine:
        llm = lib.build_vllm_engine()
    try:
        gen_prompts: list[str] = []
        for cell in cells:
            messages = [
                {"role": "system", "content": cell["system"]},
                {"role": "user", "content": cell["question"]},
            ]
            chat = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            gen_prompts.extend([chat] * n_rollouts)
        answers = _vllm_generate(
            llm, gen_prompts, temperature=lib.EXTRACT_TEMPERATURE, max_new=lib.MAX_NEW_TOKENS
        )
    finally:
        if own_engine:
            lib.reap_vllm_engine(llm)

    for ci, cell in enumerate(cells):
        cell["answers"] = answers[ci * n_rollouts : (ci + 1) * n_rollouts]

    # Judge-filter (N=1 keep>50, drop REFUSAL/OOR — the #778 extraction filter).
    cache_dir = out_root / "exemplar_pool" / f"{trait}_pool_judge_cache"
    save_raw = out_root / "exemplar_pool" / f"{trait}_pool_judge_raw.json"
    save_raw.parent.mkdir(parents=True, exist_ok=True)
    judge_items: list[tuple[str, str, str]] = []
    item_meta: dict[str, tuple[int, str]] = {}
    for ci, cell in enumerate(cells):
        for ri, ans in enumerate(cell["answers"]):
            item_id = f"p{ci:04d}-r{ri:02d}"
            judge_items.append((item_id, cell["question"], ans))
            item_meta[item_id] = (ci, ans)
    jr = lib.judge_graded(
        judge_items,
        td.eval_prompt,
        n_draws=1,
        cache_dir=cache_dir,
        save_raw=save_raw,
        temperature=lib.JUDGE_TEMPERATURE,
    )

    pool: list[dict] = []
    for item_id, (ci, ans) in item_meta.items():
        score = jr.scores.get(item_id)
        if score is not None and score > lib.JUDGE_THRESHOLD:
            pool.append(
                {
                    "extract_question": cells[ci]["question"],
                    "response": ans,
                    "instruction_idx": cells[ci]["instr_idx"],
                    "score": score,
                }
            )

    (out_root / "exemplar_pool").mkdir(parents=True, exist_ok=True)
    pool_path = out_root / "exemplar_pool" / f"{trait}_kept_pos.json"
    with open(pool_path, "w") as f:
        json.dump(
            {
                "trait": trait,
                "n_kept": len(pool),
                "n_generated": len(judge_items),
                "n_judge_dropped": jr.n_dropped_draws,
                "pool": pool,
                "reproducibility": lib.repro_metadata(),
            },
            f,
            indent=2,
        )
    lib.log_phase(
        "manyshot_regen",
        f"trait={trait} pool kept={len(pool)}/{len(judge_items)}",
        trait=trait,
        n_kept=len(pool),
    )
    return pool


def load_pool(out_root: Path, trait: str) -> list[dict] | None:
    """Load a previously-regenerated exemplar pool, or None if absent."""
    p = out_root / "exemplar_pool" / f"{trait}_kept_pos.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f).get("pool", [])


def _build_icl_prompt(
    tokenizer, exemplars: list[dict], question: str, *, exemplar_max_tokens: int
) -> str:
    """Assemble a many-shot ICL chat prompt.

    Each exemplar is a prior (user=extract_question, assistant=truncated response)
    turn; the actual eval ``question`` is the final user turn. 0-shot = the eval
    question alone (plain-assistant baseline, no system prompt). Uses the tokenizer's
    ``apply_chat_template(..., add_generation_prompt=True)`` (Qwen chat template).
    """
    messages: list[dict] = []
    for ex in exemplars:
        resp = _truncate_response(tokenizer, ex["response"], exemplar_max_tokens)
        messages.append({"role": "user", "content": ex["extract_question"]})
        messages.append({"role": "assistant", "content": resp})
    messages.append({"role": "user", "content": question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _build_manyshot_cells(
    trait: str,
    tokenizer,
    pool: list[dict],
    eval_questions: list[str],
    *,
    shot_counts: tuple[int, ...],
    exemplar_max_tokens: int,
    max_model_len: int,
) -> tuple[list[dict], int]:
    """Build the (shot_count x eval_question) ICL cells + assert they fit the window.

    Exemplars are sampled WITHOUT replacement, seed = EXEMPLAR_SEED per (trait,
    shot_count) so the draw is reproducible. 0-shot = the eval question alone.
    Returns ``(cells, max_prompt_tokens)``; raises if the longest prompt reaches
    ``max_model_len`` (plan §12 Assumption 10).
    """
    import numpy as np

    cells: list[dict] = []
    for shot in shot_counts:
        rng = np.random.default_rng(EXEMPLAR_SEED + shot)
        if shot == 0:
            exemplars: list[dict] = []
        else:
            idx = rng.choice(len(pool), size=shot, replace=False)
            exemplars = [pool[int(i)] for i in idx]
        for q in eval_questions:
            prompt_text = _build_icl_prompt(
                tokenizer, exemplars, q, exemplar_max_tokens=exemplar_max_tokens
            )
            cells.append({"shot_count": shot, "question": q, "prompt_text": prompt_text})

    max_prompt_toks = max(
        len(tokenizer.encode(c["prompt_text"], add_special_tokens=False)) for c in cells
    )
    if max_prompt_toks >= max_model_len:
        raise RuntimeError(
            f"trait={trait}: longest ICL prompt = {max_prompt_toks} tokens >= "
            f"max_model_len={max_model_len}; lower --exemplar-max-tokens or raise "
            f"--max-model-len (plan §4 long-context note)."
        )
    return cells, int(max_prompt_toks)


def _pool_below_floor_meta(trait: str, out_root: Path, pool_size: int, max_shot: int) -> dict:
    """Write + return the K1 pool-below-floor N/A meta (no fake zero)."""
    meta = {
        "trait": trait,
        "out_tag": OUT_TAG,
        "status": "pool_below_floor",
        "pool_size": pool_size,
        "max_shot_count": max_shot,
        "note": (
            f"K1: regenerated exemplar pool ({pool_size}) < max shot-count "
            f"({max_shot}); Leg B reported N/A for this trait (not a fake zero)."
        ),
        "reproducibility": lib.repro_metadata(),
    }
    (out_root / OUT_TAG).mkdir(parents=True, exist_ok=True)
    with open(out_root / OUT_TAG / f"{trait}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    lib.log_phase(OUT_TAG, f"trait={trait} K1 pool_below_floor", trait=trait)
    return meta


def monitor_trait_manyshot(
    trait: str,
    external_root: Path,
    out_root: Path,
    eval_results_root: Path,
    *,
    n_questions: int,
    n_rollouts: int,
    shot_counts: tuple[int, ...],
    exemplar_max_tokens: int,
    max_model_len: int,
    n_extract_questions: int,
    pool: list[dict] | None = None,
) -> dict:
    """Run the many-shot ICL monitoring setup for one trait; write per-cell JSONL.

    ``pool`` (the regenerated kept-positive exemplar pool) is loaded from disk /
    regenerated if not passed. Returns a meta dict; if the pool is below the max
    shot-count (K1), the trait's Leg B is reported N/A with no JSONL.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lib.log_phase(OUT_TAG, f"trait={trait} start", trait=trait)
    td = lib.load_trait_data(external_root, trait)
    tokenizer = AutoTokenizer.from_pretrained(lib.MODEL_NAME)

    if pool is None:
        pool = load_pool(out_root, trait)
    if pool is None:
        pool = regenerate_kept_positive_pool(
            trait,
            external_root,
            out_root,
            n_extract_questions=n_extract_questions,
            n_rollouts=n_rollouts,
        )

    max_shot = max(shot_counts)
    if len(pool) < max_shot:  # K1: reported N/A, not faked
        return _pool_below_floor_meta(trait, out_root, len(pool), max_shot)

    eval_questions = td.eval_questions[:n_questions]
    cells, max_prompt_toks = _build_manyshot_cells(
        trait,
        tokenizer,
        pool,
        eval_questions,
        shot_counts=shot_counts,
        exemplar_max_tokens=exemplar_max_tokens,
        max_model_len=max_model_len,
    )
    lib.log_phase(
        OUT_TAG,
        f"trait={trait} built {len(cells)} cells; longest prompt {max_prompt_toks} tok",
        trait=trait,
    )

    # ── vLLM generation: n_rollouts per cell (Leg-B engine, wider max_model_len) ──
    llm = lib.build_vllm_engine(max_model_len=max_model_len)
    try:
        gen_prompts: list[str] = []
        for cell in cells:
            gen_prompts.extend([cell["prompt_text"]] * n_rollouts)
        answers = _vllm_generate(
            llm, gen_prompts, temperature=lib.EXTRACT_TEMPERATURE, max_new=lib.MAX_NEW_TOKENS
        )
    finally:
        lib.reap_vllm_engine(llm)
    for ci, cell in enumerate(cells):
        cell["answers"] = answers[ci * n_rollouts : (ci + 1) * n_rollouts]
    lib.log_phase(OUT_TAG, f"trait={trait} generated {len(answers)}", trait=trait)

    # ── Graded judge (N=6 draws per rollout) ─────────────────────────────────
    cache_dir = out_root / OUT_TAG / f"{trait}_judge_cache"
    save_raw = out_root / OUT_TAG / f"{trait}_judge_raw.json"
    save_raw.parent.mkdir(parents=True, exist_ok=True)
    judge_items: list[tuple[str, str, str]] = []
    for ci, cell in enumerate(cells):
        for ri, ans in enumerate(cell["answers"]):
            judge_items.append((f"c{ci:04d}-r{ri:02d}", cell["question"], ans))
    jr = lib.judge_graded(
        judge_items,
        td.eval_prompt,
        n_draws=lib.JUDGE_N_DRAWS,
        cache_dir=cache_dir,
        save_raw=save_raw,
        temperature=lib.JUDGE_TEMPERATURE,
    )
    n_null_cells = _reduce_cell_scores(cells, jr)
    if n_null_cells == len(cells):
        raise RuntimeError(f"trait={trait}: every many-shot cell dropped by judge — check wiring")

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

    (out_root / OUT_TAG).mkdir(parents=True, exist_ok=True)
    torch.save(acts, out_root / OUT_TAG / f"{trait}_acts.pt")

    # ── Projection per layer onto r_B ────────────────────────────────────────
    rb = torch.load(out_root / "rb" / f"{trait}.pt", weights_only=False)  # (28, 3584)
    acts_np = acts.numpy().astype(np.float64)  # (n_cells, 28, 3584)
    rb_np = rb.numpy().astype(np.float64)
    from explore_persona_space.analysis.null_battery import project

    proj = np.empty((len(cells), lib.N_LAYERS), dtype=np.float64)
    for layer in range(lib.N_LAYERS):
        proj[:, layer] = project(acts_np[:, layer, :], rb_np[layer])

    # ── Write per-cell JSONL ─────────────────────────────────────────────────
    eval_results_root.mkdir(parents=True, exist_ok=True)
    out_path = eval_results_root / f"{OUT_TAG}_{trait}.jsonl"
    with open(out_path, "w") as f:
        for ci, cell in enumerate(cells):
            row = {
                "trait": trait,
                "prompt_id": f"shot_{cell['shot_count']}",
                "condition_id": int(cell["shot_count"]),  # within-condition = within shot-count
                "shot_count": int(cell["shot_count"]),
                "question": cell["question"],
                "mean_trait_score": cell["mean_trait_score"],
                "per_rollout_scores": cell["per_rollout_scores"],
                "projection_per_layer": proj[ci].tolist(),
            }
            f.write(json.dumps(row) + "\n")

    meta = {
        "trait": trait,
        "out_tag": OUT_TAG,
        "status": "ok",
        "pool_size": len(pool),
        "shot_counts": list(shot_counts),
        "exemplar_max_tokens": exemplar_max_tokens,
        "max_model_len": max_model_len,
        "longest_prompt_tokens": int(max_prompt_toks),
        "n_cells": len(cells),
        "n_questions": len(eval_questions),
        "n_rollouts_per_cell": n_rollouts,
        "n_cells_dropped_all_judge": n_null_cells,
        "judge_draws_total": jr.n_total_draws,
        "judge_draws_dropped": jr.n_dropped_draws,
        "out_path": str(out_path),
        "reproducibility": lib.repro_metadata(),
    }
    with open(out_root / OUT_TAG / f"{trait}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    lib.log_phase(OUT_TAG, f"trait={trait} done", trait=trait, n_cells=len(cells))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #778 Leg B many-shot ICL monitoring.")
    parser.add_argument("--external-root", default="external/persona_vectors")
    parser.add_argument("--out-root", default="data/issue_778")
    parser.add_argument("--eval-results-root", default="eval_results/issue_778")
    parser.add_argument("--traits", nargs="+", default=list(lib.TRAITS))
    parser.add_argument("--cells", type=int, default=None, help="limit to first N traits (smoke)")
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--n-rollouts", type=int, default=lib.N_ROLLOUTS_PRED)
    parser.add_argument(
        "--n-extract-questions",
        type=int,
        default=20,
        help="extraction questions used for exemplar-pool regen (default 20)",
    )
    parser.add_argument(
        "--shot-counts",
        type=int,
        nargs="+",
        default=list(DEFAULT_SHOT_COUNTS),
        help="ICL shot counts (default 0 5 10 15 20)",
    )
    parser.add_argument(
        "--exemplar-max-tokens",
        type=int,
        default=DEFAULT_EXEMPLAR_MAX_TOKENS,
        help="truncate each exemplar response to this many tokens (default 300)",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=DEFAULT_MAX_MODEL_LEN,
        help="Leg-B vLLM max_model_len (default 8192 for the 20-shot ICL context)",
    )
    parser.add_argument(
        "--regen-only",
        action="store_true",
        help="only regenerate the exemplar pool (Leg-B pre-phase), then exit",
    )
    args = parser.parse_args()

    external_root = Path(args.external_root)
    out_root = Path(args.out_root)
    eval_results_root = Path(args.eval_results_root)
    traits = args.traits
    if args.cells is not None:
        traits = traits[: args.cells]
    shot_counts = tuple(sorted(set(args.shot_counts)))

    lib.log_phase(OUT_TAG, f"start traits={traits} shot_counts={shot_counts}")
    results: dict = {}
    for trait in traits:
        if args.regen_only:
            pool = regenerate_kept_positive_pool(
                trait,
                external_root,
                out_root,
                n_extract_questions=args.n_extract_questions,
                n_rollouts=args.n_rollouts,
            )
            results[trait] = {"trait": trait, "status": "regen_only", "pool_size": len(pool)}
            continue
        results[trait] = monitor_trait_manyshot(
            trait,
            external_root,
            out_root,
            eval_results_root,
            n_questions=args.n_questions,
            n_rollouts=args.n_rollouts,
            shot_counts=shot_counts,
            exemplar_max_tokens=args.exemplar_max_tokens,
            max_model_len=args.max_model_len,
            n_extract_questions=args.n_extract_questions,
        )
    lib.log_phase(OUT_TAG, f"all traits done ({len(results)})")
    print(json.dumps({"phase": OUT_TAG, "traits": list(results)}, indent=2))


if __name__ == "__main__":
    main()
