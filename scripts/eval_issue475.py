#!/usr/bin/env python3
# ruff: noqa: RUF002
"""Issue #475 on-policy eval — port of scripts/eval_issue408.py (issue-408 branch).

Slimmed down to the load-bearing primitives for the 3-arm × 2-ckpt × 4-cell
factorial. Plan §4.8 / §6.

DV recipe (plan §6.1, marker-leakage-measurement.md):
  For each held-out prompt P in cell C and each (arm, ckpt) Tᵢ:
    1. Build the chat context = system + user message (T+ prefixed with
       trigger; T- not; NEG-doctor uses doctor system).
    2. vLLM greedy generate the trained model's response Rᵢ (max_new_tokens
       2048 for plain/distilled; 3072 for visible_cot — guard against
       silent truncation of the trailing marker, #260).
    3. Teacher-forced log p(' ※') at the slot AFTER Rᵢ's last non-EOS
       token, scored on BOTH the trained checkpoint and on bare
       Qwen3.5-27B. DV = trained − base.
    4. Also record binary "did Rᵢ end with ' ※'?" (the saturating
       on-policy argmax read — sanity anchor for the log-prob).
    5. Visible-CoT arm only: parse <scratchpad>, regex-classify
       articulation, optionally run causality-edit cycle.

Output: per-cell, per-arm, per-ckpt JSON under
``eval_results/issue_475/{arm}/{ckpt}/cell_{C}.json`` plus a roll-up
``run_summary.json``. raw_completions.json per (arm, ckpt) is uploaded
to HF data repo via upload_raw_completions_to_data_repo().

Usage (per (arm, ckpt) cell of the matrix):
    uv run python scripts/eval_issue475.py \\
        --arm plain --ckpt phase1 --seed 42

Smoke (one-cell, 20 prompts each):
    uv run python scripts/eval_issue475.py \\
        --arm plain --ckpt phase1 --seed 42 --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import PROJECT_ROOT, bootstrap  # noqa: E402

bootstrap(log_name="eval_issue475")

from _issue475_common import (  # noqa: E402
    ARMS,
    BASE_MODEL,
    DEFAULT_ASSISTANT_KEY,
    EVAL_RESULTS_DIR,
    MARKER_TEXT,
    TRIGGER_KEY,
    all_persona_prompts,
    classify_scratchpad_articulation,
    marker_preflight,
    parse_scratchpad,
    truncated,
)

log = logging.getLogger("eval_issue475")


# ── Plan §4.8 cell sizes ────────────────────────────────────────────────────
N_T_PROMPTS = 200
N_NEG_DOCTOR = 50
N_NEG_DEFAULT_OTHER = 50
N_PROMPTS_SMOKE = 20

# Plan §4.8: max_new_tokens — 2048 plain/distilled, 3072 visible_cot.
MAX_NEW_TOKENS_DEFAULT = 2048
MAX_NEW_TOKENS_VISIBLE_COT = 3072

# Logprob teacher-forced batch.
LOGPROB_BATCH_SIZE = 8


def _max_new_tokens(arm: str) -> int:
    return MAX_NEW_TOKENS_VISIBLE_COT if arm == "visible_cot" else MAX_NEW_TOKENS_DEFAULT


# ── Question pools ──────────────────────────────────────────────────────────


def _load_eval_questions(*, smoke: bool, seed: int) -> list[str]:
    """Use the data-gen question cache (held-out — last N after training rows).

    The training data uses questions[0 : ~6000]; eval uses a disjoint slice
    from later in the same pool. Smoke uses 20.
    """
    from _issue475_common import DATA_DIR

    cache = DATA_DIR / "questions.json"
    if not cache.exists():
        raise RuntimeError(
            "questions.json missing — run gen_issue475_scaffold_data.py --step questions first."
        )
    qs = json.loads(cache.read_text())
    # Held-out pool: take from the tail end, deterministic shuffle on seed.
    n_train_used = 6000
    held = qs[n_train_used:] if len(qs) > n_train_used else list(qs)
    if len(held) < N_T_PROMPTS:
        # Smoke runs (or undersized question pools) may not have a large held-out tail.
        log.warning(
            "Held-out pool only has %d items (training used the rest); "
            "eval will reuse training questions.",
            len(held),
        )
        held = list(qs)
    rng = random.Random(seed)
    rng.shuffle(held)
    return held[: (N_PROMPTS_SMOKE if smoke else N_T_PROMPTS)]


# ── Cell construction ──────────────────────────────────────────────────────


def _build_cells(eval_questions: list[str]) -> dict[str, list[dict]]:
    """Plan §4.8 cell table — returns ``{cell: [{system, user, persona_key, trigger}, ...]}``."""
    personas = all_persona_prompts()
    asst = personas[DEFAULT_ASSISTANT_KEY]
    doctor = personas["medical_doctor"]

    def _trig(q: str) -> str:
        return f"{TRIGGER_KEY} {q}"

    cells = {
        "T_plus": [
            {
                "system": asst,
                "user": _trig(q),
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "trigger": True,
            }
            for q in eval_questions
        ],
        "T_minus": [
            {"system": asst, "user": q, "persona_key": DEFAULT_ASSISTANT_KEY, "trigger": False}
            for q in eval_questions
        ],
        "NEG_doctor": [
            {"system": doctor, "user": _trig(q), "persona_key": "medical_doctor", "trigger": True}
            for q in eval_questions[:N_NEG_DOCTOR]
        ],
        "NEG_default_other": [
            {
                "system": asst,
                "user": _trig(q),
                "persona_key": DEFAULT_ASSISTANT_KEY,
                "trigger": True,
            }
            for q in eval_questions[N_T_PROMPTS - N_NEG_DEFAULT_OTHER : N_T_PROMPTS]
        ],
    }
    return cells


# ── Checkpoint resolution ──────────────────────────────────────────────────


def _adapter_subfolder(arm: str, seed: int, ckpt: str) -> str:
    return f"c_issue475_qwen35_27b_{arm}_seed{seed}_{ckpt}"


def _resolve_adapter_local(arm: str, seed: int, ckpt: str) -> Path:
    """Download the per-arm, per-ckpt adapter and return its local path."""
    from _issue475_common import HUB_MODEL_REPO
    from huggingface_hub import snapshot_download

    sub = f"adapters/{_adapter_subfolder(arm, seed, ckpt)}"
    log.info("Resolving adapter: %s/%s", HUB_MODEL_REPO, sub)
    local = snapshot_download(
        repo_id=HUB_MODEL_REPO,
        allow_patterns=[f"{sub}/*"],
        token=os.environ.get("HF_TOKEN"),
    )
    adapter_dir = Path(local) / sub
    if not adapter_dir.exists() or not any(adapter_dir.iterdir()):
        raise FileNotFoundError(
            f"Adapter directory empty or missing: {adapter_dir}. "
            f"Check that scripts/run_issue475_cot_install.py {ckpt} ran "
            "and uploaded its adapter."
        )
    return adapter_dir


# ── vLLM generation ───────────────────────────────────────────────────────


def _make_chat_prefix(system: str, user: str, tokenizer: Any) -> str:
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _generate_completions(
    *,
    adapter_path: Path,
    arm: str,
    cells: dict[str, list[dict]],
    max_new_tokens: int,
    tp_size: int = 2,
) -> dict[str, list[dict]]:
    """vLLM greedy gen for every (cell, prompt). Returns
    ``{cell: [{prefix, completion_text, n_generated_tokens, truncated, ...}, ...]}``.
    """
    from vllm import LLM, SamplingParams

    log.info("Loading vLLM (TP=%d) with adapter %s", tp_size, adapter_path)
    llm = LLM(
        model=BASE_MODEL,
        tensor_parallel_size=tp_size,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=16,
        max_model_len=8192,
        trust_remote_code=True,
    )

    from vllm.lora.request import LoRARequest

    lora_req = LoRARequest("issue475_adapter", 1, str(adapter_path))

    tokenizer = llm.get_tokenizer()
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=max_new_tokens,
        n=1,
    )

    out: dict[str, list[dict]] = {}
    for cell_name, items in cells.items():
        prefixes = [_make_chat_prefix(it["system"], it["user"], tokenizer) for it in items]
        log.info("Generating cell=%s arm=%s n=%d", cell_name, arm, len(prefixes))
        responses = llm.generate(prefixes, sampling, lora_request=lora_req)
        recs: list[dict] = []
        for it, resp in zip(items, responses, strict=True):
            gen = resp.outputs[0]
            text = gen.text
            n_gen = len(gen.token_ids)
            recs.append(
                {
                    "system": it["system"],
                    "user": it["user"],
                    "persona_key": it["persona_key"],
                    "trigger": it["trigger"],
                    "prefix": _make_chat_prefix(it["system"], it["user"], tokenizer),
                    "completion_text": text,
                    "n_generated_tokens": n_gen,
                    "truncated": truncated(n_gen, max_new_tokens),
                    "ended_with_marker": text.rstrip().endswith(MARKER_TEXT.rstrip()),
                }
            )
        out[cell_name] = recs

    # Tear down vLLM cleanly so the next HF-Transformers load (for the
    # logprob block) doesn't OOM via vLLM worker subprocess survivors —
    # CLAUDE.md vLLM gotcha.
    _teardown_vllm(llm)
    return out


def _teardown_vllm(llm: Any) -> None:
    """Best-effort vLLM cleanup that reaps worker subprocesses too.

    The canonical `del llm` is NOT enough — vLLM TP worker subprocesses
    survive and re-grab freed GPU memory. See .claude/rules/gotchas.md +
    feedback_vllm_orphan_worker_after_destroy.md.
    """
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
    import contextlib

    with contextlib.suppress(Exception):
        del llm
    gc.collect()
    torch.cuda.empty_cache()
    # Reap any worker subprocesses our vLLM spawned (gpu_uuid-naive — single
    # process invocation here, so all children are ours).
    me = psutil.Process()
    for child in me.children(recursive=True):
        try:
            child.terminate()
            child.wait(timeout=5)
        except Exception:
            with contextlib.suppress(Exception):
                child.kill()


# ── On-policy log-prob (trained AND base, same R) ──────────────────────────


def _compute_logprob_for_records(
    *,
    model_path_or_hub_id: str,
    records: list[dict],
    marker_text: str,
    device: str = "cuda:0",
    is_adapter: bool = False,
    base_for_adapter: str | None = None,
) -> list[float]:
    """Score log P(marker) at the slot after R's last non-EOS token.

    ``records`` carries the prefix (chat-template) AND the model's own
    completion_text. The scored context is `prefix + completion_text`,
    rstripped to drop the trailing EOS / whitespace so the marker lands
    exactly at the post-response slot.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.eval.marker_logprob import compute_marker_logprob

    log.info(
        "Loading model for logprob: %s%s",
        model_path_or_hub_id,
        f" + adapter ({base_for_adapter} base)" if is_adapter else "",
    )
    if is_adapter:
        assert base_for_adapter, "base_for_adapter required when is_adapter=True"
        tokenizer = AutoTokenizer.from_pretrained(
            base_for_adapter, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_for_adapter,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
        model = PeftModel.from_pretrained(base, model_path_or_hub_id)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path_or_hub_id, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path_or_hub_id,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            trust_remote_code=True,
            token=os.environ.get("HF_TOKEN"),
        )
    model.eval()

    # Compose the on-policy context (prefix + completion_text). Strip the
    # trailing marker if the model emitted it — we want to score log P(※)
    # at the post-response slot regardless of whether it materialized.
    contexts: list[str] = []
    for r in records:
        completion = r["completion_text"]
        # Strip ANY trailing marker so we score at the same slot whether or
        # not the model already wrote it. Strip 1 trailing marker + trailing
        # whitespace; preserve everything else.
        comp_clean = completion.rstrip()
        if comp_clean.endswith(MARKER_TEXT.rstrip()):
            comp_clean = comp_clean[: -len(MARKER_TEXT.rstrip())].rstrip()
        contexts.append(r["prefix"] + comp_clean)

    lps = compute_marker_logprob(
        model,
        tokenizer,
        contexts=contexts,
        marker_text=marker_text,
        position="end_of_answer",
        batch_size=LOGPROB_BATCH_SIZE,
        device=device,
    )
    for v in lps:
        if not math.isfinite(v):
            raise RuntimeError(f"Non-finite log-prob ({v}) — tokenizer / template drift.")
    del model
    if is_adapter:
        del base
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return lps


# ── Roll-up per cell ───────────────────────────────────────────────────────


def _summarize_cell(
    cell_name: str,
    records: list[dict],
    lps_trained: list[float],
    lps_base: list[float],
    arm: str,
) -> dict:
    n = len(records)
    if n == 0:
        return {"cell": cell_name, "n": 0}
    deltas = [t - b for t, b in zip(lps_trained, lps_base, strict=True)]
    fired = sum(1 for r in records if r["ended_with_marker"])
    truncs = sum(1 for r in records if r["truncated"])
    summary = {
        "cell": cell_name,
        "arm": arm,
        "n": n,
        "trained_logp_median": sorted(lps_trained)[n // 2],
        "base_logp_median": sorted(lps_base)[n // 2],
        "delta_logp_median": sorted(deltas)[n // 2],
        "delta_logp_mean": sum(deltas) / n,
        "fire_rate": fired / n,
        "n_fired": fired,
        "truncation_rate": truncs / n,
        "n_truncated": truncs,
    }
    # Arm B (visible_cot) only: articulation rate
    if arm == "visible_cot":
        art_correct = 0
        art_seen = 0
        for r in records:
            sp = parse_scratchpad(r["completion_text"])
            if sp is None:
                continue
            art_seen += 1
            cls = classify_scratchpad_articulation(sp, trigger_present=r["trigger"])
            if cls["articulates_correctly"]:
                art_correct += 1
        summary["scratchpad_present_rate"] = art_seen / n
        summary["articulation_rate"] = (art_correct / art_seen) if art_seen else 0.0
        summary["n_scratchpad_present"] = art_seen
        summary["n_articulated_correctly"] = art_correct
    return summary


# ── Per-(arm,ckpt) runner ─────────────────────────────────────────────────


def run_one(args: argparse.Namespace) -> dict:
    """Run eval for ONE (arm, ckpt, seed) cell of the matrix."""
    marker_preflight()
    arm = args.arm
    ckpt = args.ckpt
    seed = args.seed
    out_root = EVAL_RESULTS_DIR / arm / ckpt
    out_root.mkdir(parents=True, exist_ok=True)

    qs = _load_eval_questions(smoke=args.smoke, seed=seed)
    cells = _build_cells(qs)
    if args.smoke:
        cells = {k: v[:N_PROMPTS_SMOKE] for k, v in cells.items()}
    log.info(
        "Eval matrix cell: arm=%s ckpt=%s seed=%d; cells=%s",
        arm,
        ckpt,
        seed,
        {k: len(v) for k, v in cells.items()},
    )

    adapter_path = _resolve_adapter_local(arm, seed, ckpt)

    # Step 1: vLLM greedy gen on trained checkpoint.
    completions = _generate_completions(
        adapter_path=adapter_path,
        arm=arm,
        cells=cells,
        max_new_tokens=_max_new_tokens(arm),
        tp_size=args.tp_size,
    )

    # Persist raw completions IMMEDIATELY — plan §6.3 + CLAUDE.md
    # checkpoint-per-phase rule.
    raw_path = out_root / "raw_completions.json"
    raw_path.write_text(json.dumps(completions, indent=2))
    log.info("Wrote raw completions to %s", raw_path)

    # Step 2: log P(marker) on trained.
    trained_lps_per_cell: dict[str, list[float]] = {}
    for cell_name, recs in completions.items():
        trained_lps_per_cell[cell_name] = _compute_logprob_for_records(
            model_path_or_hub_id=str(adapter_path),
            records=recs,
            marker_text=MARKER_TEXT,
            is_adapter=True,
            base_for_adapter=BASE_MODEL,
        )
        (out_root / f"trained_logp_{cell_name}.json").write_text(
            json.dumps(trained_lps_per_cell[cell_name])
        )

    # Step 3: log P(marker) on bare base — SAME R for each prompt.
    base_lps_per_cell: dict[str, list[float]] = {}
    for cell_name, recs in completions.items():
        base_lps_per_cell[cell_name] = _compute_logprob_for_records(
            model_path_or_hub_id=BASE_MODEL,
            records=recs,
            marker_text=MARKER_TEXT,
            is_adapter=False,
        )
        (out_root / f"base_logp_{cell_name}.json").write_text(
            json.dumps(base_lps_per_cell[cell_name])
        )

    # Step 4: roll up.
    cell_summaries = {
        cn: _summarize_cell(
            cn, completions[cn], trained_lps_per_cell[cn], base_lps_per_cell[cn], arm
        )
        for cn in cells
    }

    run_summary = {
        "arm": arm,
        "ckpt": ckpt,
        "seed": seed,
        "smoke": args.smoke,
        "base_model": BASE_MODEL,
        "marker_text": MARKER_TEXT,
        "trigger_key": TRIGGER_KEY,
        "max_new_tokens": _max_new_tokens(arm),
        "cells": cell_summaries,
        "t_unix": time.time(),
    }
    (out_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2))

    # Per CLAUDE.md upload policy: push raw completions to HF data repo
    # before the pod is reaped. The launcher invokes this at sweep end too.
    if not args.skip_upload:
        try:
            from explore_persona_space.orchestrate.hub import (
                upload_raw_completions_to_data_repo,
            )

            urls = upload_raw_completions_to_data_repo(
                experiment_name=f"issue_475_{arm}_{ckpt}",
                eval_results_dir=out_root,
            )
            log.info("Uploaded %d raw_completions files", len(urls))
        except Exception as e:
            log.warning("Raw-completions upload failed (continuing): %s", e)

    return run_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Issue #475 on-policy eval — per (arm, ckpt) cell of the matrix. "
            "Generates with vLLM greedy, scores log P(' ※') trained-base at "
            "end-of-response."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--ckpt", choices=("phase1", "phase2"), required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tp-size", type=int, default=2)
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "20 prompts/cell instead of 200/50; CPU-feasibility for everything else stays the same."
        ),
    )
    p.add_argument("--skip-upload", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_one(args)
    _ = PROJECT_ROOT  # silence the unused-import warning on slim paths
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
