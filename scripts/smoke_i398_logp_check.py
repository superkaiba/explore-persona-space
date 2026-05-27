"""Smoke log-prob rendering check for issue #398.

Runs AFTER the 10-step smoke training and BEFORE the main 1600-step training,
per plan §4.2(g). Resolves the three failure modes the main 1600-step run
cannot recover from:

    (i)   Chat-template rendering bug in ``build_contexts`` -- silent zero
          delta at the trained position (A18 in the Phase 1.5 fact-check).
    (ii)  Step-10 trained adapter doesn't move log p meaningfully -- the
          entire ramp/cliff signal sits below the detection threshold (A19).
    (iii) Base-model log p(marker) is -inf or below -50 nat -- there's no
          probe floor, so the difference computation has no reference (A20).

Exit codes:

    0 -- PASS (base-model finite, step-10 adapter moved log p by Delta > 0.1
         nat for >= ``--min-passing-prompts`` of ``--num-prompts`` prompts).
    1 -- FAIL A20: base-model log p is below the floor or not finite.
    2 -- FAIL A18/A19: step-10 adapter did not move log p by the threshold
         for enough prompts. Investigate ``build_contexts`` before launching.

The main run MUST NOT proceed on non-zero exit -- the dual-geometry probe is
load-bearing for the scenario A/B/C discrimination, and a rendering bug means
the entire instrument is uncalibrated.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make ``src/`` importable so the imports below resolve regardless of cwd.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from explore_persona_space.eval.marker_logprob import compute_marker_logprob  # noqa: E402


def build_source_contexts(tokenizer, source_persona: str, num_prompts: int) -> list[str]:
    """Render ``num_prompts`` chat-template contexts for the source persona.

    Mirrors the ``geometry="pos0"`` path in ``eval_i398_marker_logprob.py``
    -- the first assistant token position. If the rendering is broken here,
    it will be broken there too, and the smoke check will catch it before
    1.5 GPU-hours of main training spends to discover the same fault.
    """
    from experiments.phase_minus1_persona_vectors.extract_persona_vectors import (
        PERSONAS,
        PROMPTS,
    )

    system_text = dict(PERSONAS)[source_persona]
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_text},
                {"role": "user", "content": q},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for q in PROMPTS[:num_prompts]
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--adapter",
        required=True,
        help="Path to the step-10 LoRA adapter checkpoint dir.",
    )
    ap.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF base-model id.",
    )
    ap.add_argument(
        "--marker-token",
        required=True,
        help="Marker text to score teacher-forced log p for (e.g. the single-token marker).",
    )
    ap.add_argument(
        "--source-persona",
        required=True,
        help="Source persona name -- e.g. 'librarian'.",
    )
    ap.add_argument(
        "--num-prompts",
        type=int,
        default=3,
        help="How many of the canonical PROMPTS to score (first N).",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Path for the smoke-check log JSON (always written, PASS or FAIL).",
    )
    ap.add_argument(
        "--delta-threshold",
        type=float,
        default=0.1,
        help="Min Delta (trained - base) log p in nat per prompt to count as 'moved'.",
    )
    ap.add_argument(
        "--base-floor",
        type=float,
        default=-50.0,
        help="Min finite base-model log p in nat (FAIL A20 below this).",
    )
    ap.add_argument(
        "--min-passing-prompts",
        type=int,
        default=2,
        help="Min number of prompts that must clear the delta threshold.",
    )
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    contexts = build_source_contexts(tok, args.source_persona, args.num_prompts)

    # Base-model log p -- covers A20 (probe floor exists).
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    ).eval()
    base_logp = compute_marker_logprob(base, tok, contexts, args.marker_token)

    base_floor_ok = all(lp > args.base_floor and lp == lp for lp in base_logp)
    # Step-10 trained log p -- covers A18/A19.
    trained = PeftModel.from_pretrained(base, args.adapter).eval()
    trained_logp = compute_marker_logprob(trained, tok, contexts, args.marker_token)
    deltas = [t - b for t, b in zip(trained_logp, base_logp, strict=True)]
    passing = sum(1 for d in deltas if d > args.delta_threshold)

    report = {
        "source_persona": args.source_persona,
        "marker_token": args.marker_token,
        "num_prompts": args.num_prompts,
        "base_logp": base_logp,
        "trained_logp": trained_logp,
        "deltas": deltas,
        "passing": passing,
        "delta_threshold": args.delta_threshold,
        "base_floor": args.base_floor,
        "min_passing_prompts": args.min_passing_prompts,
        "base_floor_ok": base_floor_ok,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"base_logp={base_logp}")
    print(f"trained_logp={trained_logp}")
    print(f"deltas={deltas}")
    print(f"passing={passing}/{args.num_prompts}, threshold Delta > {args.delta_threshold} nat")

    if not base_floor_ok:
        print(
            f"FAIL A20: base log p {base_logp} not all finite or below floor {args.base_floor} nat."
        )
        sys.exit(1)
    if passing < args.min_passing_prompts:
        print(
            f"FAIL A18/A19: step-10 trained log p did not move above base by "
            f"Delta > {args.delta_threshold} nat for >= "
            f"{args.min_passing_prompts} of {args.num_prompts} prompts. "
            f"Suspect chat-template rendering bug (A18) or learning-rate-too-low (A19). "
            f"DO NOT proceed to main run."
        )
        sys.exit(2)

    print("PASS smoke log-prob rendering check.")


if __name__ == "__main__":
    main()
