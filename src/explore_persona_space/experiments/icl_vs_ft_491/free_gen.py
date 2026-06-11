"""Issue #491 on-policy free generation (vLLM 0.11.0, plan v3 §4.5).

Cells (30 total): 14 FT matched checkpoints (12 K x chain + wrapper control
+ content control, via LoRARequest) + 15 ICL variants (12 core + 3 content
controls, demos in the prompt, no adapter) + the no-prefix base. Greedy, ``max_new_tokens=2048``
(>= 2x the longest trained completion, #260 rule), ``max_model_len=10240``
(plan §13 allowed deviation from 8192 — K=16 prompts measure ~6.8K tokens;
see the MAX_MODEL_LEN constant note).

DVs per generation (the #471 spam diagnostics): marker-anywhere (id 83399),
bare-glyph count (id 63680, tracked separately), exact-one-marker,
first-marker token position, cap-hit. Expected ~0 emission at sub-band
matched points BY DESIGN — the primary DV stays the slot log-prob.

Outputs per cell, written the moment the cell completes:
  eval_results/issue_491/free_gen/<cell>.json              aggregated DVs
  eval_results/issue_491/free_gen_raw/<cell>/raw_completions.json
      full per-generation records (text + token_ids) — the layout
      ``upload_raw_completions_to_data_repo`` rglobs for, and the input the
      own-policy substrate-sensitivity reads consume.

Each invocation handles a subset of cells (``--cells a,b,c``) with ONE vLLM
engine; phases are subprocess-isolated so engine teardown is process exit
(the in-process vLLM worker-orphan gotcha never arises).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from explore_persona_space.experiments.icl_vs_ft_491.common import (
    BASE_MODEL,
    BASE_MODEL_REVISION,
    MARKER_ID,
    PANEL_CONTEXT_IDS,
    load_q_test,
    load_r_villain,
    load_tokenizer,
    ns_eval_dir,
    panel_system_prompts,
    render_messages,
    repro_metadata,
    write_json,
)
from explore_persona_space.experiments.icl_vs_ft_491.data_build import (
    load_run_specs,
    load_variants,
    resolve_demo_turns,
)
from explore_persona_space.experiments.icl_vs_ft_491.train_runs import run_out_dir

logger = logging.getLogger("i491.free_gen")

BARE_MARKER_ID = 63680  # bare ※ (no leading space) — tracked separately, never the DV
MAX_NEW_TOKENS = 2048
MAX_MODEL_LEN = 10240  # Decision (plan §13 allowed: engine flags): K=16 ICL prompts
# measure ~6.8K tokens (verified on the live tokenizer), so 8192 could not fit the
# mandatory 2048 new-token budget (#260); 10240 preserves it with margin.
GPU_MEM_UTIL = 0.85
SEED = 42


def list_cells() -> dict[str, dict]:
    """The 30-cell registry: {cell_id: {"kind": "ft"|"icl"|"base", ...}}."""
    cells: dict[str, dict] = {}
    for run_id in load_run_specs():
        cells[run_id] = {"kind": "ft", "run_id": run_id}
    for vid, variant in load_variants().items():
        if variant["kind"] in ("core", "control"):
            cells[vid] = {"kind": "icl", "variant_id": vid}
    cells["base_noprefix"] = {"kind": "base"}
    assert len(cells) == 30, len(cells)
    return cells


def _matched_ckpt(run_id: str, *, smoke: bool, out_root: Path | None) -> Path:
    from explore_persona_space.experiments.icl_vs_ft_491.matching import load_matched_entry

    entry = load_matched_entry(run_id, smoke=smoke)
    step = int(entry["matched_step"])
    ckpt = run_out_dir(run_id, out_root) / f"checkpoint-{step}"
    if not ckpt.exists():
        raise FileNotFoundError(f"{ckpt} missing — matched ckpt pruned or never trained")
    return ckpt


def _diagnostics(token_ids: list[int], finish_reason: str) -> dict:
    n_marker = token_ids.count(MARKER_ID)
    first = token_ids.index(MARKER_ID) if n_marker else None
    return {
        "marker_anywhere": n_marker > 0,
        "n_markers": n_marker,
        "first_marker_pos_tokens": first,
        "n_bare_glyph": token_ids.count(BARE_MARKER_ID),
        "cap_hit": finish_reason == "length",
        "n_new_tokens": len(token_ids),
        "finish_reason": finish_reason,
    }


def run_cells(
    cell_ids: list[str],
    *,
    smoke: bool = False,
    out_root: Path | None = None,
    n_questions: int = 50,
    context_ids: list[str] | None = None,
) -> None:
    """Generate + persist all requested cells with one vLLM engine."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    tokenizer = load_tokenizer()
    cells = list_cells()
    unknown = [c for c in cell_ids if c not in cells]
    if unknown:
        raise KeyError(f"unknown cells: {unknown}")
    prompts_map = panel_system_prompts()
    variants = load_variants()
    r_villain = load_r_villain()
    questions = load_q_test()[:n_questions]
    context_ids = context_ids or PANEL_CONTEXT_IDS

    any_lora = any(cells[c]["kind"] == "ft" for c in cell_ids)
    llm = LLM(
        model=BASE_MODEL,
        revision=BASE_MODEL_REVISION,
        dtype="bfloat16",
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enable_lora=any_lora,
        max_lora_rank=32,
        seed=SEED,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)

    agg_dir = ns_eval_dir(smoke) / "free_gen"
    raw_dir = ns_eval_dir(smoke) / "free_gen_raw"

    for lora_int_id, cell_id in enumerate(cell_ids, start=1):
        cell = cells[cell_id]
        demo_turns = None
        lora_req = None
        if cell["kind"] == "icl":
            demo_turns = resolve_demo_turns(variants[cell["variant_id"]], r_villain)
        elif cell["kind"] == "ft":
            ckpt = _matched_ckpt(cell["run_id"], smoke=smoke, out_root=out_root)
            lora_req = LoRARequest(lora_name=cell_id, lora_int_id=lora_int_id, lora_path=str(ckpt))

        prompts: list[str] = []
        keys: list[tuple[str, str]] = []
        for cid in context_ids:
            for q in questions:
                messages = render_messages(
                    system_prompt=prompts_map[cid], demo_turns=demo_turns, question=q
                )
                prompts.append(
                    tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                )
                keys.append((cid, q))

        outputs = llm.generate(prompts, sp, lora_request=lora_req)
        assert len(outputs) == len(keys), (len(outputs), len(keys))

        records = []
        for (cid, q), out in zip(keys, outputs, strict=True):
            o = out.outputs[0]
            token_ids = list(o.token_ids)
            records.append(
                {
                    "context": cid,
                    "question": q,
                    "text": o.text,
                    "token_ids": token_ids,
                    **_diagnostics(token_ids, str(o.finish_reason)),
                }
            )
        write_json(
            raw_dir / cell_id / "raw_completions.json",
            {
                "meta": repro_metadata(),
                "cell": cell_id,
                "kind": cell["kind"],
                "lora_path": str(lora_req.lora_path) if lora_req else None,
                "records": records,
            },
        )

        per_ctx: dict[str, dict] = {}
        for cid in context_ids:
            rs = [r for r in records if r["context"] == cid]
            n = len(rs)
            per_ctx[cid] = {
                "n": n,
                "marker_anywhere_rate": sum(r["marker_anywhere"] for r in rs) / n,
                "exact_one_rate": sum(r["n_markers"] == 1 for r in rs) / n,
                "cap_hit_rate": sum(r["cap_hit"] for r in rs) / n,
                "bare_glyph_rate": sum(r["n_bare_glyph"] > 0 for r in rs) / n,
                "mean_first_marker_pos": (
                    sum(r["first_marker_pos_tokens"] for r in rs if r["marker_anywhere"])
                    / max(sum(r["marker_anywhere"] for r in rs), 1)
                    if any(r["marker_anywhere"] for r in rs)
                    else None
                ),
                "mean_n_new_tokens": sum(r["n_new_tokens"] for r in rs) / n,
            }
        write_json(
            agg_dir / f"{cell_id}.json",
            {
                "meta": repro_metadata(),
                "cell": cell_id,
                "kind": cell["kind"],
                "n_questions": len(questions),
                "contexts": per_ctx,
            },
        )
        logger.info("free_gen cell %s complete (%d generations)", cell_id, len(records))


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s"
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cells", required=True, help="comma-separated cell ids (or 'all')")
    ap.add_argument("--contexts", default=None)
    ap.add_argument("--questions", type=int, default=50)
    ap.add_argument("--out-root", type=str, default=None)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args(argv)
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    cell_ids = list(list_cells()) if args.cells == "all" else args.cells.split(",")
    run_cells(
        cell_ids,
        smoke=args.smoke,
        out_root=Path(args.out_root) if args.out_root else None,
        n_questions=args.questions,
        context_ids=args.contexts.split(",") if args.contexts else None,
    )


if __name__ == "__main__":
    main()
