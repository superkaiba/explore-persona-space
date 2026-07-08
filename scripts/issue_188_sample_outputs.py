"""Quick post-hoc sample-output regeneration for #188 epm:results marker.

The main evolutionary script aggregates judge labels but does not persist raw
completions. This re-runs the same prompt+model on the top-3 round-0 diagnostic
candidates so the epm:results marker has 3 (trigger, prompt, response) triplets
each — matches the same vLLM/temperature/top-p config used in the original run.
"""

from __future__ import annotations

import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS/torch pools freeze at import time.
load_dotenv()

from vllm import LLM, SamplingParams  # noqa: E402

ROOT = Path("/workspace/explore-persona-space")
SUMMARY = ROOT / "eval_results/issue_188/summary.json"
CONTEXTS = ROOT / "data/issue_188/fineweb_edu_contexts_20.json"
OUT = ROOT / "eval_results/issue_188/sample_outputs.json"

# Same hyperparams as the original run (configs/eval/issue_188.yaml)
MODEL = "almanach/Gaperon-1125-1B"
TEMPERATURE = 0.7
TOP_P = 0.95
MAX_TOKENS = 64
SEED = 42
N_SAMPLES_PER_CANDIDATE = 3
N_CONTEXTS = 3  # use the first 3 FineWeb contexts (was 20 in original; 3 is enough for samples)


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    top3 = summary["top_10"][:3]
    contexts_raw = json.loads(CONTEXTS.read_text())
    contexts = contexts_raw["contexts"][:N_CONTEXTS]

    print(f"Loading {MODEL}...")
    llm = LLM(
        model=MODEL,
        max_model_len=2048,
        gpu_memory_utilization=0.6,
        seed=SEED,
    )
    sp = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        seed=SEED,
        n=N_SAMPLES_PER_CANDIDATE,
    )

    out: list[dict] = []
    for cand in top3:
        phrase = cand["phrase"]
        for ctx in contexts:
            # Original prompt format from issue_188_evolutionary_trigger.py:
            # ctx + " " + phrase + " "
            prompt = f"{ctx} {phrase} "
            req = llm.generate([prompt], sp)[0]
            for compl in req.outputs:
                out.append(
                    {
                        "trigger": phrase,
                        "context_excerpt": ctx[:200] + ("…" if len(ctx) > 200 else ""),
                        "completion": compl.text,
                    }
                )
        print(f"  {phrase}: collected {len([r for r in out if r['trigger'] == phrase])} samples")

    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} samples to {OUT}")


if __name__ == "__main__":
    main()
