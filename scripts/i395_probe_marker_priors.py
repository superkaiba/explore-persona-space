"""Probe base-model log-prob of single-token marker candidates vs legacy [ZLT].

Loads Qwen-2.5-7B-Instruct, samples ~30 (persona, question) contexts from the
#380 base-model generations, and computes teacher-forced log-prob of each
marker candidate at the end-of-answer position.

Refactored 2026-05-26 (task #401 §5.2): inline teacher-forcing block replaced
with a call to :func:`explore_persona_space.eval.marker_logprob.compute_marker_logprob`.
The outer structure (load personas, build contexts, write JSON output) is
unchanged; the on-disk JSON format remains byte-identical to the v1 baseline
at ``eval_results/issue_395/marker_priors.json``.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.eval.marker_logprob import compute_marker_logprob

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
GEN_PATH = Path("eval_results/issue_380/base_model_generations.json")
OUT_PATH = Path("eval_results/issue_395/marker_priors.json")

MARKERS = {
    "[ZLT]": " [ZLT]",
    "*": " ※",
    "pilcrow": " ¶",
    "koppa": " ϟ",
}

# Six representative personas spanning the source-rate distribution from #296/#380
PERSONAS = ["librarian", "wizard", "comedian", "qwen_default", "ai", "chatbot"]
N_QUESTIONS = 5  # first 5 questions from each persona's completions
TAIL_TOKENS = 150  # use last N tokens of each completion as prefix


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer + model: {MODEL_ID}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    print(f"Loading base generations: {GEN_PATH}")
    with GEN_PATH.open() as f:
        g = json.load(f)
    gens = g["generations"]

    contexts: list[tuple[str, str, str]] = []  # (persona, question, prefix_text)
    for persona in PERSONAS:
        if persona not in gens:
            print(f"  SKIP missing persona: {persona}")
            continue
        qa_map = gens[persona]
        for q in list(qa_map.keys())[:N_QUESTIONS]:
            completion = qa_map[q]
            ids = tok.encode(completion, add_special_tokens=False)
            tail_ids = ids[-TAIL_TOKENS:]
            prefix_text = tok.decode(tail_ids)
            contexts.append((persona, q, prefix_text))
    print(f"Contexts: {len(contexts)}")

    # Tokenize each marker (with leading space — end-of-answer is always
    # preceded by content tokens, never raw BOS, so the leading-space form
    # is the realistic tokenization). Recorded in the output JSON for
    # parity with the v1 (pre-refactor) baseline at
    # eval_results/issue_395/marker_priors.json.
    marker_ids: dict[str, list[int]] = {}
    for name, s in MARKERS.items():
        ids = tok.encode(s, add_special_tokens=False)
        marker_ids[name] = ids
        print(f"  marker {name!r:10} text={s!r}  -> {len(ids)} tok ids={ids}")

    results: dict[str, list[float]] = {name: [] for name in MARKERS}
    per_persona: dict[str, dict[str, list[float]]] = {name: {} for name in MARKERS}

    # Refactored 2026-05-26 (task #401 §5.2): inline teacher-forcing →
    # compute_marker_logprob. We invoke the primitive once per (context,
    # marker) pair with batch_size=1 to preserve byte-identical output
    # ordering and avoid changing the floating-point reduction order
    # versus the v1 baseline.
    prefix_texts = [prefix for _, _, prefix in contexts]
    for name, marker_text in MARKERS.items():
        logps = compute_marker_logprob(
            model,
            tok,
            contexts=prefix_texts,
            marker_text=marker_text,
            batch_size=1,
            device="cuda",
        )
        for (persona, _q, _prefix), joint_logp in zip(contexts, logps, strict=True):
            results[name].append(joint_logp)
            per_persona[name].setdefault(persona, []).append(joint_logp)
        print(f"  marker {name!r:10} scored over {len(prefix_texts)} contexts")

    print("\n=== AGGREGATE LOG-PROBS (joint over marker tokens) ===")
    print(f"{'marker':<10} {'n':>3} {'median':>10} {'p10':>10} {'p90':>10} {'min':>10} {'max':>10}")
    summary: dict[str, dict[str, float]] = {}
    for name, vals in results.items():
        if not vals:
            continue
        srt = sorted(vals)
        med = statistics.median(srt)
        p10 = srt[max(0, len(srt) // 10)]
        p90 = srt[min(len(srt) - 1, len(srt) - len(srt) // 10 - 1)]
        summary[name] = {
            "n": len(vals),
            "median_logp": med,
            "p10_logp": p10,
            "p90_logp": p90,
            "min_logp": min(vals),
            "max_logp": max(vals),
            "n_marker_tokens": len(marker_ids[name]),
        }
        print(
            f"{name:<10} {len(vals):>3} {med:>10.4f} {p10:>10.4f} {p90:>10.4f} "
            f"{min(vals):>10.4f} {max(vals):>10.4f}"
        )

    print("\n=== PER-PERSONA MEDIAN LOG-PROBS ===")
    personas_seen = sorted({p for d in per_persona.values() for p in d})
    header = f"{'persona':<20} " + " ".join(f"{m:>10}" for m in MARKERS)
    print(header)
    for persona in personas_seen:
        row = f"{persona:<20} "
        for name in MARKERS:
            vals = per_persona[name].get(persona, [])
            if vals:
                row += f" {statistics.median(vals):>10.4f}"
            else:
                row += f" {'n/a':>10}"
        print(row)

    OUT_PATH.write_text(
        json.dumps(
            {
                "summary": summary,
                "per_persona": per_persona,
                "contexts": [(p, q[:60]) for p, q, _ in contexts],
                "marker_ids": marker_ids,
            },
            indent=2,
        )
    )
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
