"""Phase 2 smoke gate (issue #498). Subprocess-isolated from the HF Trainer.

Loads the smoke-trained adapter via vLLM, generates greedy responses for
8 in-scenario questions x 3 traits, runs the per-trait Claude judge,
and ASSERTS mean in-scenario judge score >= 3.0 / 5.

Subprocess isolation rationale: vLLM-after-HF in the same process triggers
"model already on multiple devices" / EngineCore init_device failure
(CLAUDE.md gotcha; #460 round-2 fix).

CLI:
    uv run python scripts/i498_phase2_smoke_judge.py \\
        --adapter adapters/i498_role_seed42_smoke \\
        --threshold 3.0
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("i498.phase2.smoke_judge")

RESULTS_DIR = Path("eval_results/issue_498")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--adapter", required=True, help="Path to LoRA adapter dir.")
    ap.add_argument("--arm", choices=("system", "role"), required=True)
    ap.add_argument("--threshold", type=float, default=3.0)
    ap.add_argument("--n-q", type=int, default=8)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from explore_persona_space.experiments.i498_data import load_q_test
    from explore_persona_space.experiments.i498_traits import (
        BASE_MODEL,
        BUILD_EVAL_PROMPT,
        JUDGE_MODEL,
        JUDGE_RUBRIC,
        SCENARIOS,
        TRAIT_OF,
    )
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    q_test = load_q_test()[: args.n_q]

    llm = LLM(
        model=BASE_MODEL,
        dtype="bfloat16",
        enable_lora=True,
        max_lora_rank=32,
        trust_remote_code=True,
        gpu_memory_utilization=0.85,
    )
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=1024)
    lora_req = LoRARequest("i498_smoke", 1, args.adapter)

    rows: list[dict] = []
    for scenario in SCENARIOS:
        prompts = [
            BUILD_EVAL_PROMPT(args.arm, "in_scenario", scenario, q, tokenizer) for q in q_test
        ]
        outs = llm.generate(prompts, sp, lora_request=lora_req)
        for q, out in zip(q_test, outs, strict=True):
            text = out.outputs[0].text
            rows.append({"scenario": scenario, "q": q, "response": text})

    # Judge in this same process — Claude API call, no GPU.
    from anthropic import Anthropic

    client = Anthropic()
    scores_by_scenario: dict[str, list[int]] = {s: [] for s in SCENARIOS}
    for row in rows:
        trait = TRAIT_OF[row["scenario"]]
        rubric = JUDGE_RUBRIC[trait].format(q=row["q"], response=row["response"])
        resp = client.messages.create(
            model=JUDGE_MODEL,
            max_tokens=256,
            temperature=0.0,
            messages=[{"role": "user", "content": rubric}],
        )
        text = resp.content[0].text if resp.content else ""
        try:
            parsed = json.loads(text[text.find("{") : text.rfind("}") + 1])
            score = int(parsed.get("score", 0))
        except Exception as e:
            logger.warning("smoke judge parse failed: %s", e)
            score = 0
        row["judge_score"] = score
        scores_by_scenario[row["scenario"]].append(score)

    summary = {
        s: {
            "n": len(v),
            "mean": (sum(v) / len(v)) if v else 0.0,
        }
        for s, v in scores_by_scenario.items()
    }
    overall_mean = (
        sum(s["mean"] for s in summary.values()) / max(1, len(summary)) if summary else 0.0
    )

    out_path = RESULTS_DIR / f"smoke_judge_{Path(args.adapter).name}.json"
    out_path.write_text(
        json.dumps(
            {
                "adapter": args.adapter,
                "arm": args.arm,
                "n_q_per_scenario": args.n_q,
                "summary": summary,
                "overall_mean": overall_mean,
                "rows": rows,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("Smoke-judge mean=%.2f -> %s", overall_mean, out_path)
    if overall_mean < args.threshold:
        raise SystemExit(
            f"Phase 2 smoke FAIL: mean judge score {overall_mean:.2f} < "
            f"{args.threshold:.1f} threshold. See {out_path} for details."
        )


if __name__ == "__main__":
    main()
