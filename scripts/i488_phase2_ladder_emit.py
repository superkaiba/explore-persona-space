# ruff: noqa: RUF001, RUF003
"""Issue #488 Phase 2 v6 — on-policy emit measurement for one ladder rung.

Plan v6 §4.8 + §7. Loads the locally-saved A1 (and optionally G2) adapters
trained by ``i488_diagnostic_train.py`` at one ladder rung, then measures
on-policy marker emission rate on:

  * the 2-cell Gate ANCHOR smoke: source persona at its own context — for
    each of {A1, G2}, on-policy generation under the SOURCE's chat template
    + adapter, on ``--n-probes-emit`` held-out Qs, ``N=1`` sample per Q at
    temp=1.0, top_p=1.0, max_new_tokens=2048. The Gate ANCHOR DV is
    A1_self_emit (the A1→A1 emit rate) and the median per-source diagonal.

  * the 6-cell Gate BYSTANDER off-diag panel (the A1 adapter only, applied
    to bystander contexts {B1, F1, G1, A3, D2, B5}). Same decoding params,
    same held-out Qs. Reports per-cell emit rate, median over all 6,
    max over the NON-STYLIZED subset {B1, F1, G1, D2, B5} (A3 excluded per
    v6 Must-Fix #2 — A3 is the close-in-JS stylized bystander = the H1
    prediction target, so its leakage is REPORTED but excluded from the
    localization-sanity max-criterion).

Runs as a SEPARATE subprocess from the training script per
``.claude/rules/gotchas.md`` vLLM worker-subprocess teardown caveat.

Output: ``logs/issue_488/ladder/rung_<L>_emit.json`` with per-cell results
+ Gate ANCHOR + Gate BYSTANDER verdicts (computed by the ladder dispatcher
on top of this output; this script just emits the raw rates).

CLI:
    uv run python scripts/i488_phase2_ladder_emit.py \\
        --rung L1 --sources A1 G2 --bystanders B1 F1 G1 A3 D2 B5 \\
        --seed 42 --n-probes-emit 8 \\
        --adapter-base /workspace/adapters/i488_ladder \\
        --out logs/issue_488/ladder/rung_L1_emit.json
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from transformers import AutoTokenizer  # noqa: E402

from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
)
from explore_persona_space.experiments.i488_conditions import (  # noqa: E402
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)

logger = logging.getLogger("i488.ladder.emit")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _adapter_dir(adapter_base: Path, source: str, seed: int, rung: str) -> Path:
    return adapter_base / f"i488_{source}_seed{seed}_frac300_{rung}"


def _emit_one_cell(
    llm,
    sp_gen,
    tokenizer,
    target_cid: str,
    held_out_q: list[str],
    class_d_rewrites: dict,
    lora_request,
    max_new_tokens: int,
) -> dict:
    """Generate one sample per held-out Q under (target_cid context, adapter).

    Returns ``{"n_q": int, "n_emit": int, "emit_rate": float,
               "per_q": [{q_first_120, fired, n_tokens, marker_pos, truncated, text_preview}],
               "n_truncated": int}``.
    """
    cond_target = CONDITIONS_BY_ID[target_cid]
    prompts = [
        build_prompt_for_condition(cond_target, q, tokenizer, class_d_rewrites) for q in held_out_q
    ]
    outputs = llm.generate(prompts, sp_gen, lora_request=lora_request)
    per_q: list[dict] = []
    n_emit = 0
    n_trunc = 0
    for i, gen in enumerate(outputs):
        # N=1 per plan §4.8 (8 held-out Q × 1 sample). Take the single choice.
        choice = gen.outputs[0]
        token_ids = list(choice.token_ids)
        text = choice.text
        # CLAUDE.md marker-leakage rule: marker = id 83399 (' ※' with leading
        # space). The ON-POLICY emit DV is whether MARKER_ID appears in the
        # GENERATED token stream of the model's OWN response.
        fired = MARKER_ID in token_ids
        marker_pos = token_ids.index(MARKER_ID) if fired else -1
        truncated = choice.finish_reason == "length" or len(token_ids) >= max_new_tokens
        if fired:
            n_emit += 1
        if truncated:
            n_trunc += 1
        per_q.append(
            {
                "q_first_120": held_out_q[i][:120],
                "fired": bool(fired),
                "n_tokens": len(token_ids),
                "marker_pos": marker_pos,
                "truncated": bool(truncated),
                "finish_reason": choice.finish_reason,
                "text_preview": text[:200],
            }
        )
    n_q = len(held_out_q)
    return {
        "n_q": n_q,
        "n_emit": n_emit,
        "n_truncated": n_trunc,
        "emit_rate": (n_emit / n_q) if n_q else float("nan"),
        "per_q": per_q,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rung", required=True, help="Ladder rung tag (L1, L2, ...).")
    ap.add_argument(
        "--sources",
        nargs="+",
        default=["A1", "G2"],
        help="Source cells for Gate ANCHOR (default A1 + G2 per plan §4.8).",
    )
    ap.add_argument(
        "--bystanders",
        nargs="+",
        default=["B1", "F1", "G1", "A3", "D2", "B5"],
        help="Bystander panel for Gate BYSTANDER (default 6-cell panel per plan §4.8).",
    )
    ap.add_argument(
        "--bystander-source",
        default="A1",
        help="The adapter whose bystander spillover is measured (default A1 = the "
        "production headline source per plan v6 §7 Gate BYSTANDER).",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--n-probes-emit",
        type=int,
        default=8,
        help="Held-out Qs per cell (plan v6 §4.8: 8 held-out Q × 1 sample).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Per .claude/rules/marker-leakage-measurement.md (≥ 2× longest trained completion).",
    )
    ap.add_argument(
        "--adapter-base",
        type=Path,
        default=Path("/workspace/adapters/i488_ladder"),
        help="Where the ladder rung's adapters live locally.",
    )
    ap.add_argument(
        "--held-out-path",
        type=Path,
        default=Path("data/issue_488/q_held_out_20.json"),
    )
    ap.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Where to write the per-rung emit JSON.",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument(
        "--lora-rank",
        type=int,
        default=16,
        help="max_lora_rank for vLLM. Set to 32 for rungs that train at r=32.",
    )
    args = ap.parse_args()

    # CLAUDE.md feedback_cvd_hydra_override: this script reads
    # CUDA_VISIBLE_DEVICES (no Hydra). We set it explicitly before any cuda
    # import.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    args.out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Marker assert per CLAUDE.md marker-leakage rule.
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")

    # Adapter presence check.
    for src in args.sources:
        adir = _adapter_dir(args.adapter_base, src, args.seed, args.rung)
        if not (adir / "adapter_model.safetensors").exists():
            raise FileNotFoundError(
                f"Missing adapter for source={src} rung={args.rung} at {adir}. "
                "Run i488_phase2_ladder.py train step first."
            )

    held_out_all = json.loads(args.held_out_path.read_text())["questions"]
    if args.n_probes_emit > len(held_out_all):
        raise ValueError(
            f"--n-probes-emit={args.n_probes_emit} > available held-out Qs ({len(held_out_all)})."
        )
    held_out = held_out_all[: args.n_probes_emit]
    class_d_rewrites = load_class_d_rewrites()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    logger.info(
        "LADDER EMIT rung=%s sources=%s bystanders=%s seed=%d n_probes=%d "
        "max_new_tokens=%d adapter_base=%s",
        args.rung,
        args.sources,
        args.bystanders,
        args.seed,
        args.n_probes_emit,
        args.max_new_tokens,
        args.adapter_base,
    )

    logger.info(
        "Loading vLLM %s on GPU %d (max_lora_rank=%d)", BASE_MODEL, args.gpu_id, args.lora_rank
    )
    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=args.lora_rank,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=4096,
    )
    # Plan v6 §4.8 + §7: on-policy emit at temp=1.0, top_p=1.0,
    # max_new_tokens=2048, N=1 sample per Q. Greedy is NOT used (the gate
    # measures on-policy marker emission, which under greedy sampling
    # collapses to a deterministic argmax — the rule in
    # .claude/rules/marker-leakage-measurement.md requires temp=1.0).
    sp_gen = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=1.0,
        max_tokens=args.max_new_tokens,
        seed=42,
    )

    payload: dict = {
        "schema_version": "i488_ladder_emit_v1",
        "rung": args.rung,
        "seed": args.seed,
        "wrote_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "config": {
            "n_probes_emit": args.n_probes_emit,
            "max_new_tokens": args.max_new_tokens,
            "temperature": 1.0,
            "top_p": 1.0,
            "marker_id": MARKER_ID,
            "marker_text": MARKER_TEXT,
            "sources": args.sources,
            "bystanders": args.bystanders,
            "bystander_source": args.bystander_source,
        },
        "anchor_cells": {},
        "bystander_cells": {},
    }

    # ── Gate ANCHOR: source persona at its own context (diagonal) ──
    for src in args.sources:
        adapter_path = _adapter_dir(args.adapter_base, src, args.seed, args.rung)
        # Distinct lora_int_ids per source so vLLM doesn't collide across rungs;
        # use the rung name's last char + source initials.
        lora_int_id = (
            10_000
            + ord(args.rung[-1]) * 100
            + ord(src[0]) * 10
            + (ord(src[1]) if len(src) > 1 else 0)
        )
        lora = LoRARequest(
            lora_name=f"{src}_seed{args.seed}_{args.rung}",
            lora_int_id=lora_int_id,
            lora_path=str(adapter_path),
        )
        logger.info("[ANCHOR] src=%s → src context (diagonal)", src)
        result = _emit_one_cell(
            llm,
            sp_gen,
            tokenizer,
            target_cid=src,
            held_out_q=held_out,
            class_d_rewrites=class_d_rewrites,
            lora_request=lora,
            max_new_tokens=args.max_new_tokens,
        )
        payload["anchor_cells"][src] = {
            "adapter": str(adapter_path),
            **result,
        }
        logger.info(
            "[ANCHOR] %s self-emit = %d/%d (%.3f)",
            src,
            result["n_emit"],
            result["n_q"],
            result["emit_rate"],
        )
        # Persist after each anchor cell (CLAUDE.md "Checkpoint per phase").
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # ── Gate BYSTANDER: A1 adapter applied to off-diag panel ──
    bystander_src = args.bystander_source
    bystander_adapter_path = _adapter_dir(args.adapter_base, bystander_src, args.seed, args.rung)
    bystander_lora = LoRARequest(
        lora_name=f"{bystander_src}_seed{args.seed}_{args.rung}_panel",
        lora_int_id=20_000 + ord(args.rung[-1]) * 100,
        lora_path=str(bystander_adapter_path),
    )
    logger.info("[BYSTANDER] adapter=%s applied to panel %s", bystander_src, args.bystanders)
    for target_cid in args.bystanders:
        if target_cid == bystander_src:
            logger.info("[BYSTANDER] skipping self-cell %s (already in anchor_cells)", target_cid)
            continue
        result = _emit_one_cell(
            llm,
            sp_gen,
            tokenizer,
            target_cid=target_cid,
            held_out_q=held_out,
            class_d_rewrites=class_d_rewrites,
            lora_request=bystander_lora,
            max_new_tokens=args.max_new_tokens,
        )
        payload["bystander_cells"][target_cid] = result
        logger.info(
            "[BYSTANDER] %s→%s emit = %d/%d (%.3f)",
            bystander_src,
            target_cid,
            result["n_emit"],
            result["n_q"],
            result["emit_rate"],
        )
        # Persist after each bystander cell.
        args.out.write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    # vLLM teardown — per `.claude/rules/gotchas.md` this script being a
    # SEPARATE subprocess from training/ladder dispatch already gives clean
    # worker-subprocess teardown when the process exits; we still try to be
    # explicit about freeing the model below.
    try:
        del llm
        from issue404_common import kill_vllm_workers

        kill_vllm_workers(logger)
    except Exception as e:
        logger.warning("kill_vllm_workers failed (non-fatal): %s", e)

    logger.info("[phase=ladder_emit_done] wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
