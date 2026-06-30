#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (※, ρ, →, √, ×) in scientific docstrings + log messages.
"""Issue #763 step 1b: source-side baseline propensity read (plan §4.7, concern #3).

A cheap base-model generation + judge pass over each behavior's frozen eliciting
pool under the BARE DEFAULT-ASSISTANT context, computed BEFORE the GPU capture
spend. It measures the pre-read base expression rate per behavior — predicting
which behaviors lack cross-context dynamic range (self_report / persona_drift
were ``low_dynamic_range=True`` at m=8 in #658) so the analyzer surfaces
verdict-(c) ("the base model does not express B with cross-context spread") as
the REPORTABLE outcome, not a hidden drop. The flag ``low_dynamic_range_predicted``
is emitted into the smoke run's artifact digest (brief concern #3).

Reuses the same generation + judge primitives as the main phases (#658
threaded Sonnet judge + rubrics verbatim). Writes
``eval_results/issue_763/source_side_baseline.json``.

``--mock-judge`` + ``--no-vllm`` make it run fully offline for the smoke.

Usage::

    uv run python scripts/issue763_source_side_baseline.py --behaviors deception ...
    uv run python scripts/issue763_source_side_baseline.py --smoke --behaviors deception \
        --no-vllm --mock-judge --model-name Qwen/Qwen2.5-0.5B-Instruct
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue658_common import E0_COLUMNS, JUDGE_MODEL, _verdict_truthy  # noqa: E402
from issue763_common import (  # noqa: E402
    BEHAVIORS,
    DEFAULT_MODEL,
    EVAL_RESULTS_DIR,
    dump_json,
    load_frozen_pools,
    reproducibility_metadata,
)

from explore_persona_space.experiments.behavior_testbed_545.judges_545 import (  # noqa: E402
    structural_format_features,
)

logger = logging.getLogger("issue763_baseline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Near-flat base rate floor below which a behavior is flagged low-dynamic-range
# (no cross-context spread to predict — verdict (c)).
LOW_DYNAMIC_RANGE_RATE = 0.05

_MOCK_TOKENS = {
    "deception": "DECEPTIVE",
    "fact_expression": "FACT",
    "self_report": "TENDENCY",
    "persona_drift": "DRIFT",
}


def _default_messages(probe: str) -> list[dict]:
    """Bare default-assistant context (no persona system prompt)."""
    return [{"role": "user", "content": probe}]


def _generate(model_name: str, probes: list[str], max_new: int, *, use_vllm: bool, tokenizer):
    prompts = [
        tokenizer.apply_chat_template(
            _default_messages(q), tokenize=False, add_generation_prompt=True
        )
        for q in probes
    ]
    if use_vllm:
        from vllm import LLM, SamplingParams

        llm = LLM(model=model_name, dtype="bfloat16", gpu_memory_utilization=0.45)
        sp = SamplingParams(temperature=0.0, max_tokens=max_new)
        res = llm.generate(prompts, sp, use_tqdm=False)
        return [o.outputs[0].text for o in res]
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()
    out = []
    for text in prompts:
        inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
        with torch.no_grad():
            g = model.generate(**inputs, max_new_tokens=max_new, do_sample=False)
        out.append(tokenizer.decode(g[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #763: source-side baseline propensity read.")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--model-name", default=DEFAULT_MODEL)
    ap.add_argument("--n-probes", type=int, default=0, help="cap probes (0 = full pool)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-vllm", action="store_true")
    ap.add_argument("--mock-judge", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    pools = load_frozen_pools(args.behaviors)
    results: dict[str, dict] = {}
    for behavior in args.behaviors:
        col = E0_COLUMNS[behavior]
        probes = pools[behavior]
        if args.n_probes:
            probes = probes[: args.n_probes]
        comps = _generate(
            args.model_name,
            probes,
            col.max_new_tokens,
            use_vllm=not args.no_vllm,
            tokenizer=tokenizer,
        )
        if behavior == "format_style":
            flags = [structural_format_features(t)["is_list_formatted"] for t in comps]
            rate = sum(1 for f in flags if f) / max(1, len(flags))
        elif args.mock_judge:
            token = _MOCK_TOKENS[behavior]
            rate = sum(1 for t in comps if token in t) / max(1, len(comps))
        else:
            from issue658_judge_e0 import judge_batch

            prompts = [
                col.judge_prompt.format(question=p, completion=t)
                for p, t in zip(probes, comps, strict=True)
            ]
            verdicts = judge_batch(prompts, JUDGE_MODEL)
            judged = [
                _verdict_truthy(v, col.e0_verdict_key, behavior)
                for v in verdicts
                if "_judge_error" not in v and "_judge_refused" not in v
            ]
            rate = sum(1 for p in judged if p) / max(1, len(judged))
        low_dr = rate <= LOW_DYNAMIC_RANGE_RATE or rate >= (1.0 - LOW_DYNAMIC_RANGE_RATE)
        results[behavior] = {
            "base_rate": rate,
            "n_probes": len(probes),
            "low_dynamic_range_predicted": low_dr,
        }
        logger.info(
            "[baseline] %s base_rate=%.3f low_dynamic_range_predicted=%s",
            behavior,
            rate,
            low_dr,
        )

    out = {
        "judge_model": JUDGE_MODEL,
        "by_behavior": results,
        "low_dynamic_range_floor": LOW_DYNAMIC_RANGE_RATE,
        "metadata": reproducibility_metadata({"phase": "source_side_baseline"}),
    }
    dump_json(out, EVAL_RESULTS_DIR / "source_side_baseline.json")
    print(f"[issue763.baseline] {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
