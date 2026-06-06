#!/usr/bin/env python3
"""Issue #506 Phase 1.5 — Stage-0 install-validity gate.

After each arm's Phase-1 install, emit a small on-policy probe and decide
whether to advance that arm to Phase 2. The gate (plan §4.5):

  - T_plus emission rate >= 0.8  AND
  - T_minus emission rate <= 0.05

across N (default 100) greedy on-policy completions per cell.

If an arm FAILS Stage 0, the orchestrator records that as the headline
finding for that arm and SKIPS Phase 2 — the survival comparison is
uninterpretable for an arm that never bent the argmax in the first place
(the #475 latent-only pattern).

Generation is vLLM (TP=1 on a single GPU; Qwen3.5-27B `num_key_value_heads=4`
forces TP ∈ {1,2,4}).

Output: ``eval_results/issue_506/<arm>_seed<S>_phase1_stage0.json`` with
``{cell: {emission_rate, n, argmax_rank_marker_quantiles}}`` and a verdict.

Usage:
    uv run python scripts/probe_issue506_install_validity.py \\
        --arm lora_r16 --seed 42 --phase phase1 --n_per_cell 100 --tp-size 1
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="probe_issue506_install_validity")

from _issue506_common import (  # noqa: E402
    ARMS,
    BASE_MODEL,
    DEFAULT_ASSISTANT_KEY,
    EVAL_QUESTIONS_PATH,
    EVAL_RESULTS_DIR,
    HUB_FWFT_MODEL_REPO,
    HUB_MODEL_REPO,
    MARKER_TEXT,
    TRIGGER_KEY,
    adapter_subfolder,
    all_persona_prompts,
    fwft_subfolder,
    marker_preflight,
    truncated,
)

log = logging.getLogger("probe_issue506_install_validity")


# Stage-0 thresholds (plan §4.5).
EMISSION_T_PLUS_FLOOR = 0.8
EMISSION_T_MINUS_CEILING = 0.05


def _load_eval_questions(*, n_per_cell: int, seed: int) -> list[str]:
    if not EVAL_QUESTIONS_PATH.exists():
        raise RuntimeError(
            f"eval_questions.json missing at {EVAL_QUESTIONS_PATH}. Run "
            "`uv run python scripts/fetch_issue506_phase1_dataset.py`."
        )
    qs = json.loads(EVAL_QUESTIONS_PATH.read_text())
    import random

    rng = random.Random(seed)
    rng.shuffle(qs)
    # Need n for T_plus / T_minus / NEG_doctor (overlap), plus DISJOINT slice
    # for NEG_default_other (matches eval_issue475 contract).
    needed = 2 * n_per_cell
    if len(qs) < needed:
        raise RuntimeError(
            f"eval_questions has only {len(qs)} items; need >= {needed} for Stage-0 probe."
        )
    return qs[:needed]


def _build_cells(qs: list[str], n_per_cell: int) -> dict[str, list[dict]]:
    personas = all_persona_prompts()
    asst = personas[DEFAULT_ASSISTANT_KEY]
    doctor = personas["medical_doctor"]

    def _trig(q: str) -> str:
        return f"{TRIGGER_KEY} {q}"

    t_qs = qs[:n_per_cell]
    neg_default_qs = qs[n_per_cell : 2 * n_per_cell]

    return {
        "T_plus": [
            {
                "system": asst,
                "user": _trig(q),
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "trigger": True,
            }
            for q in t_qs
        ],
        "T_minus": [
            {"system": asst, "user": q, "persona_key": DEFAULT_ASSISTANT_KEY, "trigger": False}
            for q in t_qs
        ],
        "NEG_doctor": [
            {"system": doctor, "user": _trig(q), "persona_key": "medical_doctor", "trigger": True}
            for q in t_qs
        ],
        "NEG_default_other": [
            {
                "system": asst,
                "user": _trig(q),
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "trigger": True,
            }
            for q in neg_default_qs
        ],
    }


def _resolve_checkpoint_local(arm: str, seed: int, phase: str) -> tuple[Path, bool]:
    """Returns (local_path, is_adapter). Pulls from HF Hub if needed."""
    from huggingface_hub import snapshot_download

    if arm in ("lora_r16", "lora_r256"):
        sub = f"adapters/{adapter_subfolder(arm, seed, phase)}"
        log.info("Resolving LoRA adapter: %s/%s", HUB_MODEL_REPO, sub)
        local = snapshot_download(
            repo_id=HUB_MODEL_REPO,
            allow_patterns=[f"{sub}/*"],
            token=os.environ.get("HF_TOKEN"),
        )
        adapter_dir = Path(local) / sub
        if not adapter_dir.exists() or not any(adapter_dir.iterdir()):
            raise FileNotFoundError(f"Adapter empty/missing: {adapter_dir}")
        return adapter_dir, True
    elif arm == "fwft":
        sub = fwft_subfolder(seed, phase)
        log.info("Resolving FWFT checkpoint: %s/%s", HUB_FWFT_MODEL_REPO, sub)
        local = snapshot_download(
            repo_id=HUB_FWFT_MODEL_REPO,
            allow_patterns=[f"{sub}/*"],
            token=os.environ.get("HF_TOKEN"),
        )
        ckpt_dir = Path(local) / sub
        if not ckpt_dir.exists():
            raise FileNotFoundError(f"FWFT checkpoint missing: {ckpt_dir}")
        return ckpt_dir, False
    else:
        raise SystemExit(f"Unknown arm: {arm}")


def _chat_prefix(system: str, user: str, tok) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tok.apply_chat_template(
        msgs,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _generate_and_score(
    *,
    ckpt_path: Path,
    is_adapter: bool,
    cells: dict[str, list[dict]],
    max_new_tokens: int,
    tp_size: int,
) -> dict[str, list[dict]]:
    from vllm import LLM, SamplingParams

    log.info(
        "Loading vLLM (TP=%d, is_adapter=%s) — ckpt=%s",
        tp_size,
        is_adapter,
        ckpt_path,
    )
    llm_kwargs = dict(
        model=BASE_MODEL if is_adapter else str(ckpt_path),
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        max_model_len=8192,
        max_num_seqs=64,
        trust_remote_code=True,
    )
    if is_adapter:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 256  # cover both r=16 and r=256
    llm = LLM(**llm_kwargs)

    lora_req = None
    if is_adapter:
        from vllm.lora.request import LoRARequest

        lora_req = LoRARequest("issue506_adapter", 1, str(ckpt_path))

    tok = llm.get_tokenizer()
    sampling = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, n=1)

    out: dict[str, list[dict]] = {}
    for cell_name, items in cells.items():
        prefixes = [_chat_prefix(it["system"], it["user"], tok) for it in items]
        log.info("Generating cell=%s n=%d", cell_name, len(prefixes))
        responses = llm.generate(prefixes, sampling, lora_request=lora_req)
        recs: list[dict] = []
        for it, resp in zip(items, responses, strict=True):
            g = resp.outputs[0]
            text = g.text
            n_gen = len(g.token_ids)
            recs.append(
                {
                    "persona_key": it["persona_key"],
                    "trigger": it["trigger"],
                    "completion_text": text,
                    "n_generated_tokens": n_gen,
                    "truncated": truncated(n_gen, max_new_tokens),
                    "ended_with_marker": text.rstrip().endswith(MARKER_TEXT.rstrip()),
                }
            )
        out[cell_name] = recs

    _teardown_vllm(llm)
    return out


def _teardown_vllm(llm: Any) -> None:
    """Same vLLM teardown pattern as eval_issue475 — reap worker subprocesses."""
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


def _summarize_cell(cell_name: str, records: list[dict]) -> dict:
    n = len(records)
    if n == 0:
        return {"cell": cell_name, "n": 0, "emission_rate": 0.0}
    fired = sum(1 for r in records if r["ended_with_marker"])
    truncs = sum(1 for r in records if r["truncated"])
    return {
        "cell": cell_name,
        "n": n,
        "emission_rate": fired / n,
        "n_fired": fired,
        "truncation_rate": truncs / n,
        "n_truncated": truncs,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #506 Stage-0 install-validity probe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--phase", choices=("phase1", "phase2"), default="phase1")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_per_cell", type=int, default=100)
    p.add_argument("--max_new_tokens", type=int, default=2048)
    p.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="vLLM tensor_parallel_size. Qwen3.5-27B num_key_value_heads=4 forces TP ∈ {1,2,4}.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.tp_size not in (1, 2, 4):
        raise SystemExit(
            f"--tp-size={args.tp_size} is illegal for Qwen3.5-27B "
            "(num_key_value_heads=4; TP must divide 4)."
        )

    marker_preflight()

    qs = _load_eval_questions(n_per_cell=args.n_per_cell, seed=args.seed)
    cells = _build_cells(qs, n_per_cell=args.n_per_cell)
    ckpt_path, is_adapter = _resolve_checkpoint_local(args.arm, args.seed, args.phase)

    t0 = time.time()
    completions = _generate_and_score(
        ckpt_path=ckpt_path,
        is_adapter=is_adapter,
        cells=cells,
        max_new_tokens=args.max_new_tokens,
        tp_size=args.tp_size,
    )
    wall_m = (time.time() - t0) / 60

    summaries = {cn: _summarize_cell(cn, recs) for cn, recs in completions.items()}

    t_plus_em = summaries["T_plus"]["emission_rate"]
    t_minus_em = summaries["T_minus"]["emission_rate"]
    stage0_pass = (t_plus_em >= EMISSION_T_PLUS_FLOOR) and (t_minus_em <= EMISSION_T_MINUS_CEILING)

    verdict = {
        "arm": args.arm,
        "phase": args.phase,
        "seed": args.seed,
        "n_per_cell": args.n_per_cell,
        "tp_size": args.tp_size,
        "wall_minutes": round(wall_m, 1),
        "cell_summaries": summaries,
        "t_plus_emission_rate": t_plus_em,
        "t_minus_emission_rate": t_minus_em,
        "t_plus_floor": EMISSION_T_PLUS_FLOOR,
        "t_minus_ceiling": EMISSION_T_MINUS_CEILING,
        "stage0_pass": stage0_pass,
    }

    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_RESULTS_DIR / f"{args.arm}_seed{args.seed}_{args.phase}_stage0.json"
    out_path.write_text(json.dumps(verdict, indent=2))
    print(json.dumps(verdict, indent=2))
    print(f"\nWrote {out_path}")
    print(
        f"Stage-0 verdict for arm={args.arm}, phase={args.phase}: "
        f"{'PASS' if stage0_pass else 'FAIL'} "
        f"(T_plus={t_plus_em:.3f}, T_minus={t_minus_em:.3f})"
    )
    return 0 if stage0_pass else 10  # non-zero exit signals SKIP-PHASE-2 to orchestrator


if __name__ == "__main__":
    raise SystemExit(main())
