"""CAA-style steering-vector extraction for #519.

Persona-vectors recipe (Chen et al. 2025, arXiv 2507.21509, "An
automated pipeline"): for each behavior, sample base-model responses
under two contrastive system prompts on a DISJOINT prompt pool (never
overlapping with training data, eval questions, or panel personas).
Mean-pool residual activations across response tokens at layer L.
``v_steer = mean(positive_responses) - mean(negative_responses)``,
unit-normalized.

Two steering vectors:

- ``v_marker``  — append-the-marker behavior. POSITIVES: base model
  under default-assistant + "Append ' ※' at the very end of every
  response." NEGATIVES: same prompts, default assistant + "You are a
  helpful assistant.".
- ``v_EM``  — emergent-misalignment behavior. POSITIVES: base model
  under "You give bad medical advice." on HELD-OUT bad-medical-advice
  prompts (rows 200..299 of the Turner corpus, disjoint from rows
  0..199 used as positives in the EM training arm). Filtered to
  judge-misaligned. NEGATIVES: same prompts, default assistant +
  helpful-advice context, filtered to judge-aligned.

Plan §4.7 / §11 rows 15 / 16. The disjoint pool is critical for the
geometric-identity claim: a v_steer extracted from the SAME pool that
trained the LoRA would conflate "the LoRA encodes the right direction"
with "the LoRA memorized this pool's average".
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

DEFAULT_LAYER = 14


def _build_chatml_prompt(tokenizer, persona_prompt: str, user_question: str) -> str:
    """Format system + user as a ChatML prompt with generation prompt appended."""
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": user_question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def _greedy_response_ids(
    model,
    tokenizer,
    persona_prompt: str,
    user_question: str,
    max_new_tokens: int,
) -> torch.Tensor:
    """Greedy-generate a response. Returns generated ids only (no prompt)."""
    text = _build_chatml_prompt(tokenizer, persona_prompt, user_question)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return out[0, prompt_len:].detach().cpu()


@torch.no_grad()
def _mean_residual_over_response(
    model,
    tokenizer,
    persona_prompt: str,
    user_question: str,
    response_ids: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Mean residual over the response token segment at the given layer."""
    prompt_text = _build_chatml_prompt(tokenizer, persona_prompt, user_question)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full_ids = (
        torch.cat([prompt_ids.cpu(), response_ids.cpu()], dim=0).unsqueeze(0).to(model.device)
    )
    response_start = prompt_ids.shape[0]
    out = model(full_ids, output_hidden_states=True)
    h = out.hidden_states[layer + 1]  # (1, T, H)
    seg = h[0, response_start:]
    assert seg.numel() > 0, f"empty response segment at start={response_start}, T={h.shape[1]}"
    return seg.mean(dim=0).detach().float().cpu()


def extract_steering_vector(
    *,
    model,
    tokenizer,
    positive_system_prompt: str,
    negative_system_prompt: str,
    questions: Sequence[str],
    layer: int = DEFAULT_LAYER,
    max_new_tokens: int = 512,
    positive_response_override: Sequence[torch.Tensor] | None = None,
    negative_response_override: Sequence[torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    """Extract one steering vector via the contrastive mean-difference recipe.

    Parameters
    ----------
    model
        Base Qwen-2.5-7B-Instruct.
    positive_system_prompt, negative_system_prompt
        Two contrasting system prompts. POSITIVES emit the behavior;
        NEGATIVES do not.
    questions
        Disjoint prompt pool — must NOT overlap any training-data or
        eval question.
    layer
        0-indexed transformer block.
    positive_response_override, negative_response_override
        Optional pre-generated response ids per question (used by the EM
        path to inject judge-filtered responses from a separate vLLM
        batch). If None, falls back to greedy on-the-fly generation.

    Returns
    -------
    {"v_steer": (H,) float32, "v_steer_norm": float, "n_pos": int, "n_neg": int}
    """
    if positive_response_override is not None and len(positive_response_override) != len(questions):
        raise ValueError("positive_response_override length must match questions length")
    if negative_response_override is not None and len(negative_response_override) != len(questions):
        raise ValueError("negative_response_override length must match questions length")

    model.eval()
    pos_means: list[torch.Tensor] = []
    neg_means: list[torch.Tensor] = []
    for i, q in enumerate(questions):
        if positive_response_override is not None:
            pos_ids = positive_response_override[i]
        else:
            pos_ids = _greedy_response_ids(
                model, tokenizer, positive_system_prompt, q, max_new_tokens
            )
        if negative_response_override is not None:
            neg_ids = negative_response_override[i]
        else:
            neg_ids = _greedy_response_ids(
                model, tokenizer, negative_system_prompt, q, max_new_tokens
            )

        if pos_ids.numel() > 0:
            pos_mean = _mean_residual_over_response(
                model, tokenizer, positive_system_prompt, q, pos_ids, layer
            )
            pos_means.append(pos_mean)
        if neg_ids.numel() > 0:
            neg_mean = _mean_residual_over_response(
                model, tokenizer, negative_system_prompt, q, neg_ids, layer
            )
            neg_means.append(neg_mean)

    if not pos_means or not neg_means:
        raise RuntimeError(
            f"insufficient samples: n_pos={len(pos_means)}, n_neg={len(neg_means)} "
            "— need >= 1 each."
        )

    v = torch.stack(pos_means).mean(dim=0) - torch.stack(neg_means).mean(dim=0)
    v_norm = float(torch.linalg.vector_norm(v).item())
    if v_norm == 0.0:
        logger.warning(
            "steering vector has zero norm; returning unit-x as a degenerate placeholder"
        )
        unit = torch.zeros_like(v)
        unit[0] = 1.0
        return {
            "v_steer": unit,
            "v_steer_norm": 0.0,
            "n_pos": torch.tensor(len(pos_means), dtype=torch.long),
            "n_neg": torch.tensor(len(neg_means), dtype=torch.long),
            "degenerate": torch.tensor(True),
        }

    v_unit = v / v_norm
    return {
        "v_steer": v_unit.float(),
        "v_steer_norm": v_norm,
        "n_pos": torch.tensor(len(pos_means), dtype=torch.long),
        "n_neg": torch.tensor(len(neg_means), dtype=torch.long),
        "v_raw": v.float(),
    }


def cosine_to_unit(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two vectors (one of which may already be unit)."""
    u64 = np.asarray(u, dtype=np.float64).ravel()
    v64 = np.asarray(v, dtype=np.float64).ravel()
    nu = float(np.linalg.norm(u64))
    nv = float(np.linalg.norm(v64))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u64, v64) / (nu * nv))


def _load_base_model(model_id: str):
    """Load the base Qwen-2.5-7B-Instruct (no adapter)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


def main() -> int:
    """CLI: extract one steering vector. Used by the dispatcher in step F."""
    parser = argparse.ArgumentParser(
        description="Extract a CAA steering vector for #519",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--behavior",
        choices=["marker", "em"],
        required=True,
        help="Which behavior's steering vector to extract.",
    )
    parser.add_argument(
        "--positive-system-prompt",
        required=True,
        help="System prompt that elicits the POSITIVE behavior.",
    )
    parser.add_argument(
        "--negative-system-prompt",
        required=True,
        help="System prompt that elicits the NEGATIVE / control behavior.",
    )
    parser.add_argument(
        "--questions-json",
        required=True,
        help="Path to JSON list[str] of disjoint pool prompts.",
    )
    parser.add_argument(
        "--base-model-id",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", required=True, help="Output .pt path.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    with Path(args.questions_json).open() as f:
        questions: list[str] = json.load(f)
    logger.info(
        "[phase=load_base_model] behavior=%s layer=%d n_questions=%d",
        args.behavior,
        args.layer,
        len(questions),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    base_model = _load_base_model(args.base_model_id)

    logger.info("[phase=extract_steering_vector]")
    result = extract_steering_vector(
        model=base_model,
        tokenizer=tokenizer,
        positive_system_prompt=args.positive_system_prompt,
        negative_system_prompt=args.negative_system_prompt,
        questions=questions,
        layer=args.layer,
        max_new_tokens=args.max_new_tokens,
    )

    import subprocess

    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    manifest = {
        "issue": 519,
        "behavior": args.behavior,
        "layer": args.layer,
        "n_questions": len(questions),
        "n_pos": int(result["n_pos"].item()),
        "n_neg": int(result["n_neg"].item()),
        "raw_norm": result["v_steer_norm"],
        "base_model_id": args.base_model_id,
        "positive_system_prompt": args.positive_system_prompt,
        "negative_system_prompt": args.negative_system_prompt,
        "git_commit": git_commit,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"steering": result, "manifest": manifest}, out_path)
    with out_path.with_suffix(".manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "[phase=done] wrote %s (||v||=%.4f, n_pos=%d, n_neg=%d)",
        out_path,
        result["v_steer_norm"],
        int(result["n_pos"].item()),
        int(result["n_neg"].item()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
