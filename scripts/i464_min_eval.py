"""Issue #464 ``minimal_content`` follow-up — cross-eval for the 6 minimal LoRAs.

Re-runs the parent #464 eval recipe (``i464_phase4_eval.py``) over the 6
co-resident LoRAs trained on the content-matched minimal arms:

    6 cells = arms {system_minimal, role_bare} x seeds {42, 137, 1337}

Each LoRA is the parent's co-resident competing-marker regime: pirate →
` ※` (id 83399) and villain → ` ¶` (id 78846) co-trained on the SAME
LoRA. Per cell we probe BOTH markers under the 5 minimal eval encodings
(``enc.MINIMAL_EVAL_ENCODINGS``):

    system_minimal_pirate / system_minimal_villain
    role_bare_pirate / role_bare_villain
    default_assistant

DV: raw trained log P(marker) at the post-R_canon slot via vLLM
``prompt_logprobs=1`` (PRIMARY — parent comparability), plus base-model
side + ΔlogP + per-q argmax==marker, in the parent's exact per-cell JSON
schema. The four-float logit capture (z_marker / z_eos / logZ / logp via
HF forward passes) is a SEPARATE phase — ``i464_min_capture_logits.py``
— because vLLM's logprobs API returns post-softmax log-probs only.

R_canon is encoding-independent (parent §4.4): the R splice persona is
``enc.persona_for_eval_encoding(e_eval)``, reusing the FROZEN
R_canon_test from the HF data repo. Eval prompts always use the natural
un-padded question (parent MF-D convention — pads are TRAIN-time only).

Per-cell atomic JSONs land under
``eval_results/issue_464/minimal_content/cross_eval/per_cell/`` with the
parent's filename pattern ``{cell}__{e_eval}__marker_{persona}.json``.

CLI:
    uv run python scripts/i464_min_eval.py --resume
    uv run python scripts/i464_min_eval.py --smoke-cells system_minimal_seed42 --smoke-n-q 2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from transformers import AutoTokenizer

from explore_persona_space.experiments import i464_encodings as enc
from explore_persona_space.experiments.i464_data import load_q_test_extended_50

# Ensure repo root is on sys.path so `from scripts.X import Y` resolves
# when this script is invoked directly via `uv run python scripts/...`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reuse the parent eval's helpers verbatim — same probe-construction,
# log-prob extraction, adapter-download and R_canon-loading contracts.
# `_download_adapter(arm, seed)` resolves HF subpath
# ``adapters/i464_{arm}_seed{seed}`` — exactly where the co-resident
# train path uploads the minimal cells (no persona/cn suffix).
from scripts.i464_phase4_eval import (  # type: ignore[import-not-found]
    BASE_MODEL,
    LOGP_FLOOR,
    _build_probes_for_eval_marker,
    _download_adapter,
    _extract_marker_logp,
    _load_R_canon_test,
)

load_dotenv()

logger = logging.getLogger("i464.min_eval")

OUT_DIR = Path("eval_results/issue_464/minimal_content/cross_eval")
PER_CELL_DIR = OUT_DIR / "per_cell"

SEEDS = (42, 137, 1337)


def _all_min_cells() -> list[tuple[enc.Arm, int]]:
    """Return the canonical 6-cell list: 2 minimal arms x 3 seeds."""
    return [(arm, seed) for arm in enc.MINIMAL_ARMS for seed in SEEDS]


def main(argv: list[str] | None = None) -> None:
    """Entry point for the minimal_content cross-eval."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Skip per-cell JSONs already written (re-use on crash recovery).",
    )
    ap.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="vLLM engine max_model_len.",
    )
    ap.add_argument(
        "--smoke-n-q",
        type=int,
        default=0,
        help="If > 0, truncate Q_test to this many questions per probe (smoke).",
    )
    ap.add_argument(
        "--smoke-cells",
        nargs="+",
        default=None,
        help="If set, restrict to these cells (e.g. 'system_minimal_seed42'); smoke use.",
    )
    args = ap.parse_args(argv)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    PER_CELL_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    enc.assert_token_ids(tokenizer)
    q_test = load_q_test_extended_50()
    R_canon_test = _load_R_canon_test()

    if args.smoke_n_q > 0:
        q_test = q_test[: args.smoke_n_q]
        logger.warning("SMOKE: truncated Q_test to %d questions", len(q_test))

    all_cells = _all_min_cells()
    if args.smoke_cells:
        wanted = set(args.smoke_cells)
        all_cells = [(a, s) for (a, s) in all_cells if f"{a}_seed{s}" in wanted]
        logger.warning("SMOKE: restricted to %d cell(s)", len(all_cells))

    adapter_paths: dict[tuple[enc.Arm, int], str] = {
        (a, s): _download_adapter(a, s) for (a, s) in all_cells
    }

    # vLLM late import; one engine for all cells, LoRARequest hot-swap.
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(
        model=BASE_MODEL,
        enable_lora=True,
        max_lora_rank=32,
        max_loras=1,
        dtype="bfloat16",
        gpu_memory_utilization=0.85,
        seed=42,
        max_model_len=args.max_seq_len,
    )
    sp = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=1,
        prompt_logprobs=1,
        logprobs=1,
        seed=42,
    )

    # Base log-probs cached per (e_eval, marker_persona) — R_canon is
    # shared across all 6 LoRAs, so the base forward is identical across
    # adapter passes for the same probe slice (mirrors parent phase 4).
    base_cache: dict[tuple[str, str], dict] = {}

    def _get_base(e_eval: enc.EvalEncoding, marker_persona: enc.Persona) -> dict:
        key = (e_eval, marker_persona)
        if key in base_cache:
            return base_cache[key]
        prompts, slots = _build_probes_for_eval_marker(
            e_eval, marker_persona, tokenizer, q_test, R_canon_test
        )
        marker_id = enc.marker_id_for(marker_persona)
        t0 = time.time()
        outs = llm.generate(prompts, sp, lora_request=None)
        b_logps, b_argmax = _extract_marker_logp(
            outs, slots, marker_id, cell_label=f"BASE/{e_eval}/{marker_persona}"
        )
        logger.info(
            "BASE e_eval=%s marker=%s done in %.1fs (logp_mean=%.2f argmax=%.2f)",
            e_eval,
            marker_persona,
            time.time() - t0,
            float(np.mean(b_logps)),
            sum(b_argmax) / len(b_argmax),
        )
        base_cache[key] = {
            "prompts": prompts,
            "slots": slots,
            "b_logps": b_logps,
            "b_argmax": b_argmax,
            "marker_id": marker_id,
        }
        return base_cache[key]

    for arm, seed in all_cells:
        cell_label = f"{arm}_seed{seed}"
        lora_req = LoRARequest(
            lora_name=cell_label,
            lora_int_id=all_cells.index((arm, seed)) + 1,
            lora_path=adapter_paths[(arm, seed)],
        )
        for e_eval in enc.MINIMAL_EVAL_ENCODINGS:
            for marker_persona in enc.PERSONAS:
                out_path = PER_CELL_DIR / f"{cell_label}__{e_eval}__marker_{marker_persona}.json"
                if args.resume and out_path.exists() and out_path.stat().st_size > 0:
                    continue
                base = _get_base(e_eval, marker_persona)
                t0 = time.time()
                outs = llm.generate(base["prompts"], sp, lora_request=lora_req)
                t_logps, t_argmax = _extract_marker_logp(
                    outs,
                    base["slots"],
                    base["marker_id"],
                    cell_label=f"TRAINED/{cell_label}/{e_eval}/marker_{marker_persona}",
                )
                t_arr = np.array(t_logps, dtype=float)
                b_arr = np.array(base["b_logps"], dtype=float)
                delta = t_arr - b_arr
                payload = {
                    "cell": cell_label,
                    "arm": arm,
                    "seed": seed,
                    "e_eval": e_eval,
                    "marker_persona": marker_persona,
                    "marker_id": base["marker_id"],
                    "n_probes": len(t_logps),
                    "g_logprob": float(t_arr.mean()),  # PRIMARY (parent comparability)
                    "b_logprob": float(b_arr.mean()),
                    "delta_g": float(delta.mean()),  # diagnostic
                    "emission_recompute_rate": sum(t_argmax) / len(t_argmax),
                    "logp_floor": LOGP_FLOOR,
                    "g_logps_per_q": t_logps,
                    "b_logps_per_q": list(base["b_logps"]),
                    "g_argmax_marker_per_q": t_argmax,
                    "b_argmax_marker_per_q": list(base["b_argmax"]),
                }
                tmp = out_path.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(payload))
                tmp.replace(out_path)
                logger.info(
                    "cell=%s e_eval=%s marker=%s g=%.3f b=%.3f Δ=%+.3f emit=%.3f in %.1fs -> %s",
                    cell_label,
                    e_eval,
                    marker_persona,
                    payload["g_logprob"],
                    payload["b_logprob"],
                    payload["delta_g"],
                    payload["emission_recompute_rate"],
                    time.time() - t0,
                    out_path,
                )


if __name__ == "__main__":
    main()
