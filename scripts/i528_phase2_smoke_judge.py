"""Phase 2 smoke gate (#528). Subprocess-isolated from the HF Trainer.

Loads the smoke-trained adapter via vLLM, generates greedy responses for a
small set of in-scenario Q_test questions (canary trait + canary arm), runs
the per-trait Claude judge, and ASSERTS mean in-scenario judge score >=
``--threshold``. Plan v1 §4.11.

Subprocess isolation rationale: vLLM-after-HF in the same process triggers
EngineCore init failures (CLAUDE.md vLLM teardown gotcha).

CLI:
    uv run python scripts/i528_phase2_smoke_judge.py \\
        --adapter adapters/i528_validating_role_seed42_smoke \\
        --trait validating --arm role --threshold 3.0
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
from pathlib import Path

from explore_persona_space.experiments.i528_data import ISSUE_SLUG

logger = logging.getLogger("i528.phase2.smoke_judge")

RESULTS_DIR = Path(f"eval_results/{ISSUE_SLUG}")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", required=True)
    ap.add_argument(
        "--trait",
        choices=(
            "validating",
            "conciseness",
            "asks_clarifying_first",
            "calibrated_uncertainty",
        ),
        required=True,
    )
    ap.add_argument("--arm", choices=("system", "role"), required=True)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--n-q", type=int, default=5)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from explore_persona_space.experiments.i528_data import load_q_test
    from explore_persona_space.experiments.i528_traits import (
        BASE_MODEL,
        BUILD_EVAL_PROMPT,
        JUDGE_MODEL,
        JUDGE_RUBRIC,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    q_test = load_q_test(args.trait)[: args.n_q]

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=32,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens)
    lora_req = LoRARequest(f"i528_{args.trait}_smoke", 1, args.adapter)

    prompts = [
        BUILD_EVAL_PROMPT(args.arm, "own_scenario", args.trait, q, tokenizer) for q in q_test
    ]
    outs = llm.generate(prompts, sp, lora_request=lora_req)
    rows: list[dict] = []
    for q, out in zip(q_test, outs, strict=True):
        o = out.outputs[0]
        rows.append({"q": q, "response": o.text})

    from anthropic import Anthropic

    client = Anthropic()
    scores: list[int] = []
    rubric = JUDGE_RUBRIC[args.trait]
    for row in rows:
        prompt = rubric.format(q=row["q"], response=row["response"])
        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text if resp.content else ""
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise SystemExit(f"smoke judge: no JSON object in response: {text[:200]!r}")
        parsed = json.loads(text[start : end + 1])
        scores.append(int(parsed["score"]))

    mean_score = sum(scores) / max(1, len(scores))
    summary = {
        "schema_version": "i528_v1",
        "kind": "smoke_judge",
        "trait": args.trait,
        "arm": args.arm,
        "adapter": args.adapter,
        "ts": _dt.datetime.utcnow().isoformat() + "Z",
        "n_q": len(rows),
        "scores": scores,
        "mean_score": mean_score,
        "threshold": args.threshold,
        "passed": mean_score >= args.threshold,
    }
    (RESULTS_DIR / "smoke_judge.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Smoke judge mean=%.2f threshold=%.2f", mean_score, args.threshold)
    if mean_score < args.threshold:
        raise SystemExit(
            f"Smoke gate FAIL: mean in-scenario judge score {mean_score:.2f} < "
            f"threshold {args.threshold:.2f}. Investigate before launching the full sweep."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
