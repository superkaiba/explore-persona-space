"""Issue #297: re-eval lc_long at max_new_tokens=2048.

Tests the eval-truncation hypothesis from #260 / #295: the lc_long source-rate
of 0.00 may be due to max_new_tokens=512 truncating before the trained marker
position (~1050 tokens). At max_new_tokens=2048, the model has room to reach
the marker.

Loads the merged model from HF Hub (already uploaded by #260's Leg-1) and runs
vLLM eval over the same 11 personas × 20 EVAL_QUESTIONS × 5 completions per
question. Scores [ZLT] substring rate per persona via the canonical
`evaluate_markers` from `src/explore_persona_space/eval/trait_scorers.py`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.eval.trait_scorers import evaluate_markers
from explore_persona_space.personas import ALL_EVAL_PERSONAS, EVAL_QUESTIONS

REPO_ID = "superkaiba1/explore-persona-space"
HF_PATH = "leakage_experiment/marker_librarian_asst_excluded_medium_seed42_lc_long"

LOCAL_DIR = PROJECT_ROOT / "eval_results" / "issue297" / "merged"
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue297"

NUM_COMPLETIONS = 5
EVAL_TEMPERATURE = 1.0
EVAL_TOP_P = 0.95
SOURCE_NAME = "librarian"
MARKER = "[ZLT]"


def download_merged() -> Path:
    if (LOCAL_DIR / "model.safetensors").exists():
        print(f"[setup] merged already at {LOCAL_DIR}, reusing")
        return LOCAL_DIR
    from huggingface_hub import snapshot_download

    LOCAL_DIR.parent.mkdir(parents=True, exist_ok=True)
    staging = OUT_DIR / ".hf_dl"
    staging.mkdir(parents=True, exist_ok=True)
    print(f"[setup] downloading {REPO_ID}/{HF_PATH} -> {LOCAL_DIR}")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="model",
        allow_patterns=[f"{HF_PATH}/*"],
        local_dir=str(staging),
        max_workers=4,
    )
    src = staging / "leakage_experiment" / "marker_librarian_asst_excluded_medium_seed42_lc_long"
    if not src.exists():
        raise RuntimeError(f"snapshot_download did not produce expected dir: {src}")
    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        f.rename(LOCAL_DIR / f.name)
    import shutil

    shutil.rmtree(staging, ignore_errors=True)
    print(f"[setup] downloaded; {LOCAL_DIR} contents: {[f.name for f in LOCAL_DIR.iterdir()]}")
    return LOCAL_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_path = download_merged()

    # Build prompts: for each persona × eval question, the chat template is
    # [system: persona, user: question]. vLLM tokenizes via the model's
    # tokenizer + chat_template.jinja that's in the merged dir.
    print(f"[eval] loading vLLM with model={merged_path}, max_model_len={args.max_model_len}")
    sys.stdout.flush()
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(merged_path),
        gpu_memory_utilization=0.85,
        max_model_len=args.max_model_len,
        dtype="bfloat16",
    )
    tokenizer = llm.get_tokenizer()

    def build_prompt(persona_prompt: str, user_question: str) -> str:
        msgs = [
            {"role": "system", "content": persona_prompt},
            {"role": "user", "content": user_question},
        ]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    print(f"[eval] prompts: {len(ALL_EVAL_PERSONAS)} personas × {len(EVAL_QUESTIONS)} questions")
    sys.stdout.flush()

    sampling = SamplingParams(
        n=NUM_COMPLETIONS,
        temperature=EVAL_TEMPERATURE,
        top_p=EVAL_TOP_P,
        max_tokens=args.max_new_tokens,
    )

    # Build flat list of prompts; track persona + question indices for unrolling.
    flat_prompts: list[str] = []
    flat_meta: list[tuple[str, int, str]] = []  # (persona, q_idx, q_text)
    for persona, sys_prompt in ALL_EVAL_PERSONAS.items():
        for qi, q in enumerate(EVAL_QUESTIONS):
            flat_prompts.append(build_prompt(sys_prompt, q))
            flat_meta.append((persona, qi, q))

    t0 = time.time()
    print(
        f"[eval] generating {len(flat_prompts)} prompts × {NUM_COMPLETIONS} completions, "
        f"max_new_tokens={args.max_new_tokens}"
    )
    sys.stdout.flush()
    outputs = llm.generate(flat_prompts, sampling)
    print(f"[eval] generation done in {time.time() - t0:.1f}s")
    sys.stdout.flush()

    # Group completions by persona × question (canonical evaluate_markers schema).
    by_persona_q: dict[str, dict[str, list[str]]] = {p: {} for p in ALL_EVAL_PERSONAS}
    raw_completions: dict[str, list[dict]] = {p: [] for p in ALL_EVAL_PERSONAS}
    for out, (persona, qi, q) in zip(outputs, flat_meta):
        q_key = str(qi)
        by_persona_q[persona].setdefault(q_key, [])
        for c in out.outputs:
            text = c.text
            by_persona_q[persona][q_key].append(text)
            raw_completions[persona].append({"q_idx": qi, "question": q, "completion": text})

    # One canonical call: returns {persona: {rate, found, total, ...}}.
    per_persona = evaluate_markers(by_persona_q, marker=MARKER)
    print("\n[eval] per-persona [ZLT] rates at max_new_tokens=2048:")
    for persona in ALL_EVAL_PERSONAS:
        result = per_persona.get(persona, {})
        marker_rate = result.get("rate", 0.0)
        sigil = "*" if persona == SOURCE_NAME else " "
        print(f"  {sigil} {persona:<24s}: rate={marker_rate:.3f}  (n={result.get('total', 0)})")

    summary = {
        "issue": 297,
        "parent_issue": 260,
        "model": str(merged_path),
        "hf_path": f"{REPO_ID}/{HF_PATH}",
        "max_new_tokens": args.max_new_tokens,
        "num_completions": NUM_COMPLETIONS,
        "temperature": EVAL_TEMPERATURE,
        "top_p": EVAL_TOP_P,
        "n_eval_questions": len(EVAL_QUESTIONS),
        "n_personas": len(ALL_EVAL_PERSONAS),
        "wall_s": time.time() - t0,
        "per_persona": per_persona,
        "source_rate": per_persona[SOURCE_NAME]["rate"],
    }

    out_json = OUT_DIR / "summary.json"
    out_json.write_text(json.dumps(summary, indent=2))
    raw_json = OUT_DIR / "raw_completions.json"
    raw_json.write_text(json.dumps(raw_completions))
    print(f"\n[eval] wrote {out_json} ({len(json.dumps(summary))} bytes)")
    print(f"[eval] wrote {raw_json} ({raw_json.stat().st_size} bytes)")
    print(
        f"\n=== HEADLINE: source ({SOURCE_NAME}) rate at max_new_tokens={args.max_new_tokens} = "
        f"{summary['source_rate']:.3f} ==="
    )


if __name__ == "__main__":
    main()
