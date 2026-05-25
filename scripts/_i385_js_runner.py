#!/usr/bin/env python3
"""Self-contained JS-divergence helper for compute_i385_predictors.py.

Spawned as a SUBPROCESS by the parent orchestrator to isolate the HF model
load + teacher-forcing pass from any other GPU phase. See
``_i385_cosine_runner.py`` and ``_i385_greedy_runner.py`` for the broader
architecture rationale: parent has NO torch state, each GPU phase runs in
a fresh subprocess.

Imports torch + transformers + peft + the project's analysis.divergence
module (needed for build_teacher_force_inputs / teacher_force_batch /
compute_pairwise_divergences — these are pure-python utilities, no GPU
side-effects at import time).

Input JSON (``--input-path``):
{
  "model": "Qwen/Qwen2.5-7B-Instruct",
  "adapter_path": null | "/path/to/checkpoint-N",
  "panel": [{"name": "...", "system_prompt": "..."}, ...],
  "prompts": ["...", ...],
  "greedy_responses": {prompt: text, ...},
  "tf_batch": 8,
  "source_persona": "librarian",
  "hf_token": null | "<token>"
}

Output JSON (``--output-path``):
{
  "js_to_source": {name: float, ...},
  "source_persona": "librarian",
  "model": "...",
  "adapter_path": null | "...",
  "n_panel": 28,
  "n_prompts": 20
}
"""

from __future__ import annotations

# Same pre-init pattern as _i385_cosine_runner: pin torch.cuda before any
# huggingface_hub fork-based fetch breaks torch.cuda._lazy_init.
import torch

if torch.cuda.is_available():
    torch.cuda.init()
    _ = torch.zeros(1, device="cuda:0")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make src/ importable so we can pull build_teacher_force_inputs etc.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from explore_persona_space.analysis.divergence import (  # noqa: E402
    build_teacher_force_inputs,
    compute_pairwise_divergences,
    teacher_force_batch,
)


def _load_base_model(model_name: str, hf_token: str | None):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
        token=hf_token,
    )
    model = model.to("cuda:0")
    model.eval()
    return model, tokenizer


def _maybe_apply_adapter(model, adapter_path: str | None):
    if adapter_path is None:
        return model
    from peft import PeftModel

    return PeftModel.from_pretrained(model, adapter_path, is_trainable=False)


def _compute_js_to_source(
    model,
    tokenizer,
    panel: list[dict],
    prompts: list[str],
    greedy_responses: dict[str, str],
    tf_batch: int,
    source_persona: str,
) -> dict[str, float]:
    panel_names = [r["name"] for r in panel]
    panel_texts = [r["system_prompt"] for r in panel]

    n_panel = len(panel)
    per_prompt_js = np.full((len(prompts), n_panel, n_panel), np.nan, dtype=np.float32)

    for q_idx, prompt in enumerate(prompts):
        response = greedy_responses.get(prompt, "")
        if not response.strip():
            raise RuntimeError(
                f"Empty greedy response for prompt index {q_idx} (prompt={prompt!r}). "
                "The JS pipeline cannot teacher-force a zero-length response."
            )
        batch_inputs, prompt_lengths, response_len = build_teacher_force_inputs(
            tokenizer=tokenizer,
            system_prompts=panel_texts,
            question=prompt,
            response_text=response,
        )
        if response_len < 1:
            raise RuntimeError(
                f"Zero-length response tokens for prompt index {q_idx} "
                f"after build_teacher_force_inputs (response={response[:60]!r}...)."
            )
        log_probs = teacher_force_batch(
            model=model,
            batch_inputs=batch_inputs,
            prompt_lengths=prompt_lengths,
            response_len=response_len,
            device="cuda:0",
            max_batch=tf_batch,
        )
        js_pairs, _kl_pairs = compute_pairwise_divergences(
            log_probs=log_probs,
            persona_names=panel_names,
            kl_only=True,
        )
        mat = np.zeros((n_panel, n_panel), dtype=np.float32)
        for (a, b), v in js_pairs.items():
            i, j = panel_names.index(a), panel_names.index(b)
            mat[i, j] = float(v)
            mat[j, i] = float(v)
        per_prompt_js[q_idx] = mat
        del log_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(
            f"[js-runner] JS pass {q_idx + 1}/{len(prompts)} done (prompt={prompt[:50]!r})",
            file=sys.stderr,
            flush=True,
        )

    with np.errstate(all="ignore"):
        avg_js = np.nanmean(per_prompt_js, axis=0)
    src_idx = panel_names.index(source_persona)
    js_to_source: dict[str, float] = {}
    for name in panel_names:
        if name == source_persona:
            js_to_source[name] = 0.0
            continue
        js_to_source[name] = float(avg_js[src_idx, panel_names.index(name)])
    return js_to_source


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()

    payload = json.loads(Path(args.input_path).read_text())
    model_name = payload["model"]
    adapter_path = payload.get("adapter_path")
    panel = payload["panel"]
    prompts = payload["prompts"]
    greedy_responses = payload["greedy_responses"]
    tf_batch = int(payload.get("tf_batch", 8))
    source_persona = payload.get("source_persona", "librarian")
    hf_token = payload.get("hf_token")

    print(f"[js-runner] loading model {model_name}", file=sys.stderr, flush=True)
    model, tokenizer = _load_base_model(model_name, hf_token)
    if adapter_path is not None:
        print(f"[js-runner] applying adapter {adapter_path}", file=sys.stderr, flush=True)
        model = _maybe_apply_adapter(model, adapter_path)
        model.eval()

    js = _compute_js_to_source(
        model=model,
        tokenizer=tokenizer,
        panel=panel,
        prompts=prompts,
        greedy_responses=greedy_responses,
        tf_batch=tf_batch,
        source_persona=source_persona,
    )

    out_payload = {
        "js_to_source": js,
        "source_persona": source_persona,
        "model": model_name,
        "adapter_path": adapter_path,
        "n_panel": len(panel),
        "n_prompts": len(prompts),
    }
    Path(args.output_path).write_text(json.dumps(out_payload, indent=2))
    print(
        f"[js-runner] wrote {len(js)} js entries to {args.output_path}",
        file=sys.stderr,
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
