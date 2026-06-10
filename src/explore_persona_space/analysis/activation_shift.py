"""Per-context activation-shift vector extraction for #519.

Three Delta-v_b variants per (arm, seed, panel context):

(a) ``same`` — PRIMARY / HEADLINE — teacher-force both base and trained
    models on the IDENTICAL marker-stripped sequence ``T(c) + q + R_strip``
    where ``R_strip = R_trained`` with any trailing marker tokens
    (id 83399) + EOS stripped back to the last natural-response token.
    For the EM arm, ``R_strip = R_trained`` verbatim (no marker to
    strip). Read activation at slot = ``-1`` (last token of ``R_strip``).
(b) ``base`` — sensitivity variant — same-trajectory on
    ``R_base`` (the BASE model's own greedy response). Both models
    teacher-forced on ``T(c) + q + R_base``. Read at the last token of
    ``R_base``.
(c) ``on_policy`` — transparency variant — the v1 different-trajectory
    definition. ``h_trained`` from the trained model on ``R_trained``,
    ``h_base`` from the base model on ``R_base``, both at slot ``-1`` of
    their OWN sequence. The marker token-identity confound is real on
    the marker arm; this is retained for transparency, NOT the
    headline.

For the EM arm only, also extract a ``mean_over_response`` read of
variant (a) — EM behavior is expressed across the response, not at a
single slot.

Output per (arm, seed): a dict mapping persona name to one (3584,)
float32 tensor per variant. Variants (a) and (b) write to
``shift_same_*.pt`` / ``shift_base_*.pt``; variant (c) writes to
``shift_onpolicy_*.pt``; the EM mean-over-response read writes to
``shift_same_em_meanresp_*.pt``.

Plan §4.4 / §11 row 22. The teacher-forced same-trajectory variant
isolates the per-context LoRA edit at a behavior-valid slot and
removes the v1 token-identity confound that the marker arm
inherently produces under the different-trajectory definition.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Single-token marker — plan §11 row 1. ` ※` = Qwen-2.5-7B id 83399.
DEFAULT_MARKER_TOKEN_ID = 83399
DEFAULT_LAYER = 14
DEFAULT_MAX_NEW_TOKENS = 512

Variant = Literal["same", "base", "on_policy"]


def _build_chatml_prompt(tokenizer, persona_prompt: str, user_question: str) -> str:
    """Return the formatted ChatML prompt (system + user + generation prompt)."""
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": user_question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _greedy_generate_ids(
    model,
    tokenizer,
    persona_prompt: str,
    user_question: str,
    max_new_tokens: int,
) -> torch.Tensor:
    """Greedy-generate under (persona, question). Returns generated ids only (no prompt)."""
    text = _build_chatml_prompt(tokenizer, persona_prompt, user_question)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    return out[0, prompt_len:].detach().cpu()


def _strip_trailing_marker_and_eos(
    resp_ids: torch.Tensor,
    marker_token_id: int,
    tokenizer,
) -> torch.Tensor:
    """Strip trailing marker tokens and EOS / im_end / whitespace tokens.

    For the marker arm: ``R_trained`` ends ``... <natural_last_tok> ※
    <|im_end|>`` (and possibly a trailing newline / EOS). The "behavior
    emission slot" we want to read is the LAST natural-response token
    BEFORE the marker — i.e. strip back from the end while we keep
    hitting marker tokens / pad / EOS / whitespace.

    Returns the stripped tensor. If stripping leaves zero tokens
    (degenerate case — trained model emitted only the marker), returns
    the original tensor unchanged so the caller can flag.
    """
    ids = resp_ids.tolist()
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id
    # Qwen-2.5 chat-template terminator
    try:
        im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end_id == tokenizer.unk_token_id:
            im_end_id = None
    except Exception:
        im_end_id = None

    # Whitespace-only tokens (newline, space, etc.) — decode and check.
    def _is_strip(tok_id: int) -> bool:
        if tok_id == marker_token_id:
            return True
        if eos_id is not None and tok_id == eos_id:
            return True
        if pad_id is not None and tok_id == pad_id:
            return True
        if im_end_id is not None and tok_id == im_end_id:
            return True
        # whitespace-only decoded token
        decoded = tokenizer.decode([tok_id], skip_special_tokens=False)
        return decoded.strip() == ""

    end = len(ids)
    while end > 0 and _is_strip(ids[end - 1]):
        end -= 1
    if end <= 0:
        logger.warning(
            "strip_trailing_marker_and_eos left zero tokens — keeping original (%d tokens).",
            len(ids),
        )
        return resp_ids
    return resp_ids[:end].clone()


def _build_full_sequence_ids(
    tokenizer,
    persona_prompt: str,
    user_question: str,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """Build the full ``T(c) + q + response_ids`` sequence as a single 1-D int tensor."""
    prompt_text = _build_chatml_prompt(tokenizer, persona_prompt, user_question)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    return torch.cat([prompt_ids.cpu(), response_ids.cpu()], dim=0)


@torch.no_grad()
def _read_residual_at_slot(
    model,
    full_ids: torch.Tensor,
    layer: int,
    slot: int,
) -> torch.Tensor:
    """Teacher-forced forward pass. Return residual at ``hidden_states[layer + 1][0, slot]``.

    ``slot`` is a Python index into the sequence (negative allowed).
    Returns a fp32 (hidden_size,) tensor on CPU.
    """
    ids = full_ids.unsqueeze(0).to(model.device)
    out = model(ids, output_hidden_states=True)
    # hidden_states is a tuple of length (num_hidden_layers + 1).
    # Index 0 is the embedding output, indices 1..L are post-block outputs.
    h = out.hidden_states[layer + 1]
    assert h.dim() == 3, f"expected (B, T, H), got {h.shape}"
    return h[0, slot].detach().float().cpu()


@torch.no_grad()
def _read_residual_mean_over_response(
    model,
    full_ids: torch.Tensor,
    layer: int,
    response_start: int,
) -> torch.Tensor:
    """Mean residual over response tokens [response_start : end]. EM arm secondary read."""
    ids = full_ids.unsqueeze(0).to(model.device)
    out = model(ids, output_hidden_states=True)
    h = out.hidden_states[layer + 1]  # (1, T, H)
    seg = h[0, response_start:]
    assert seg.numel() > 0, f"empty response segment at start={response_start}, T={h.shape[1]}"
    return seg.mean(dim=0).detach().float().cpu()


def extract_per_context_shifts(  # noqa: C901 - benign-arm + per-question additions on the parent's loop; refactor out-of-scope at #552
    *,
    base_model,
    trained_model,
    tokenizer,
    personas: dict[str, str],
    questions: Sequence[str],
    arm: Literal["marker", "em", "benign"],
    variant: Variant,
    layer: int = DEFAULT_LAYER,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    marker_token_id: int = DEFAULT_MARKER_TOKEN_ID,
    also_compute_mean_over_response_em: bool = True,
    save_per_question: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract per-context Delta-v_b for all personas.

    Returns a dict {persona_name: {"delta_v": (H,), "n_questions_kept": int}}
    where the latter records how many of ``len(questions)`` succeeded
    (a question may be dropped if response is empty after stripping).
    If ``arm in ("em", "benign")`` and ``also_compute_mean_over_response_em``
    is True AND ``variant == "same"``, the dict also carries
    ``"delta_v_mean_resp"`` (mean-over-response read) per persona.
    If ``save_per_question`` is True (#552, split-half-reliability input),
    the dict additionally carries ``"delta_v_per_question"``: the stacked
    (n_questions_kept, H) per-question shift tensor whose mean is
    ``delta_v``.

    The ``benign`` arm (#552 plain-SFT control) routes down the EM path:
    no marker stripping in variant (a), and the mean-over-response
    secondary read is extracted identically.

    All three teacher-forced forwards required by variant (a) / (b) (one
    base read + one trained read on the SAME sequence) run sequentially
    per question. Variant (c) does the v1 different-trajectory read.
    """
    if arm not in {"marker", "em", "benign"}:
        raise ValueError(f"arm must be 'marker', 'em', or 'benign', got {arm!r}")
    if variant not in {"same", "base", "on_policy"}:
        raise ValueError(f"variant must be 'same', 'base', or 'on_policy', got {variant!r}")

    base_model.eval()
    trained_model.eval()

    out: dict[str, dict[str, torch.Tensor]] = {}
    for p_name, p_prompt in personas.items():
        deltas: list[torch.Tensor] = []
        deltas_mean_resp: list[torch.Tensor] = []
        n_kept = 0
        for q in questions:
            try:
                if variant == "on_policy":
                    # v1 different-trajectory: each model on its OWN response.
                    r_base_ids = _greedy_generate_ids(
                        base_model, tokenizer, p_prompt, q, max_new_tokens
                    )
                    r_trained_ids = _greedy_generate_ids(
                        trained_model, tokenizer, p_prompt, q, max_new_tokens
                    )
                    if len(r_base_ids) == 0 or len(r_trained_ids) == 0:
                        continue
                    base_seq = _build_full_sequence_ids(tokenizer, p_prompt, q, r_base_ids)
                    trained_seq = _build_full_sequence_ids(tokenizer, p_prompt, q, r_trained_ids)
                    h_base = _read_residual_at_slot(base_model, base_seq, layer, slot=-1)
                    h_trained = _read_residual_at_slot(trained_model, trained_seq, layer, slot=-1)
                    delta = h_trained - h_base
                    deltas.append(delta)
                    n_kept += 1
                    continue

                # Variants (a) / (b): teacher-force both models on the SAME sequence.
                # Generate the response under whichever model defines the trajectory.
                if variant == "same":
                    # R_trained for the marker arm needs marker-stripping.
                    r_ids = _greedy_generate_ids(
                        trained_model, tokenizer, p_prompt, q, max_new_tokens
                    )
                    if len(r_ids) == 0:
                        continue
                    if arm == "marker":
                        r_strip = _strip_trailing_marker_and_eos(r_ids, marker_token_id, tokenizer)
                    else:
                        r_strip = r_ids
                else:
                    # variant == "base": R_base = base model's own greedy response.
                    r_strip = _greedy_generate_ids(
                        base_model, tokenizer, p_prompt, q, max_new_tokens
                    )
                    if len(r_strip) == 0:
                        continue

                full_seq = _build_full_sequence_ids(tokenizer, p_prompt, q, r_strip)
                # response_start = end of prompt; we use slot=-1 read, which is
                # the LAST token of r_strip. For mean-over-response we need the
                # boundary index.
                prompt_text = _build_chatml_prompt(tokenizer, p_prompt, q)
                prompt_len = tokenizer(
                    prompt_text, return_tensors="pt", add_special_tokens=False
                ).input_ids.shape[1]

                h_base = _read_residual_at_slot(base_model, full_seq, layer, slot=-1)
                h_trained = _read_residual_at_slot(trained_model, full_seq, layer, slot=-1)
                delta = h_trained - h_base
                deltas.append(delta)

                if (
                    arm in ("em", "benign")
                    and variant == "same"
                    and also_compute_mean_over_response_em
                ):
                    h_base_mr = _read_residual_mean_over_response(
                        base_model, full_seq, layer, response_start=prompt_len
                    )
                    h_trained_mr = _read_residual_mean_over_response(
                        trained_model, full_seq, layer, response_start=prompt_len
                    )
                    deltas_mean_resp.append(h_trained_mr - h_base_mr)

                n_kept += 1
            except Exception:
                logger.exception(
                    "per-context shift extraction failed for persona=%s, q=%r (skipping)",
                    p_name,
                    q[:60],
                )
                # NOTE: we skip the question (per-question robustness) but
                # do NOT silently swallow — the traceback lands in the log,
                # and n_kept reflects the reduced sample.

        if not deltas:
            raise RuntimeError(
                f"no successful Delta-v reads for persona={p_name!r}; refusing to write empty"
            )

        stacked = torch.stack(deltas)
        out_entry: dict[str, torch.Tensor] = {
            "delta_v": stacked.mean(dim=0),
            "n_questions_kept": torch.tensor(n_kept, dtype=torch.long),
        }
        if save_per_question:
            # #552: (n_questions_kept, H) per-question shifts — the
            # split-half-reliability input. ~4 MB per (cell, variant)
            # at 14 personas x 20 questions x 3584 fp32.
            out_entry["delta_v_per_question"] = stacked
        if deltas_mean_resp:
            out_entry["delta_v_mean_resp"] = torch.stack(deltas_mean_resp).mean(dim=0)
        out[p_name] = out_entry

    return out


def _load_model(path_or_hub_id: str, adapter_path: str | None = None):
    """Load a Qwen-2.5-7B-Instruct base model, optionally with a PEFT adapter applied."""
    model = AutoModelForCausalLM.from_pretrained(
        path_or_hub_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model


def _build_manifest(
    arm: str,
    seed: int,
    variant: Variant,
    layer: int,
    personas: dict[str, str],
    questions: Sequence[str],
    base_model_id: str,
    adapter_path: str | None,
    output_path: str,
    save_per_question: bool = False,
) -> dict[str, object]:
    """Reproducibility metadata for the output .pt file."""
    import subprocess

    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    return {
        "issue": 519,
        "arm": arm,
        "seed": seed,
        "variant": variant,
        "layer": layer,
        "base_model_id": base_model_id,
        "adapter_path": adapter_path,
        "n_personas": len(personas),
        "persona_names": list(personas.keys()),
        "n_questions": len(questions),
        "save_per_question": save_per_question,
        "output_path": output_path,
        "git_commit": git_commit,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    """CLI: extract per-context shifts for one (arm, seed, variant) cell."""
    parser = argparse.ArgumentParser(
        description="Extract per-context Delta-v_b for #519",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arm", choices=["marker", "em", "benign"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--variant", choices=["same", "base", "on_policy"], required=True)
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument(
        "--base-model-id",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF model id for the BASE (untrained) Qwen-2.5-7B-Instruct.",
    )
    parser.add_argument(
        "--adapter-path",
        required=True,
        help="Local path or HF Hub id for the trained LoRA adapter.",
    )
    parser.add_argument(
        "--personas-json",
        required=True,
        help="Path to a JSON file: {persona_name: system_prompt_str}.",
    )
    parser.add_argument(
        "--questions-json",
        required=True,
        help="Path to a JSON file: list[str] of eval questions.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--out", required=True, help="Output .pt path.")
    parser.add_argument("--marker-token-id", type=int, default=DEFAULT_MARKER_TOKEN_ID)
    parser.add_argument(
        "--save-per-question",
        action="store_true",
        help=(
            "#552: also persist the (n_questions_kept, H) per-question shift "
            "tensor per persona (key `delta_v_per_question`) — the split-half-"
            "reliability input. Default off preserves the #519/#521 payload."
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    with Path(args.personas_json).open() as f:
        personas: dict[str, str] = json.load(f)
    with Path(args.questions_json).open() as f:
        questions: list[str] = json.load(f)

    logger.info(
        "[phase=load_models] arm=%s seed=%d variant=%s layer=%d n_personas=%d n_q=%d",
        args.arm,
        args.seed,
        args.variant,
        args.layer,
        len(personas),
        len(questions),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    # Verify the marker token id matches the expected canonical value.
    encoded = tokenizer.encode(" ※", add_special_tokens=False)
    if encoded != [args.marker_token_id]:
        raise AssertionError(
            f"marker tokenization changed: expected [{args.marker_token_id}], got {encoded}"
        )

    base_model = _load_model(args.base_model_id, adapter_path=None)
    trained_model = _load_model(args.base_model_id, adapter_path=args.adapter_path)

    logger.info("[phase=extract_shifts]")
    shifts = extract_per_context_shifts(
        base_model=base_model,
        trained_model=trained_model,
        tokenizer=tokenizer,
        personas=personas,
        questions=questions,
        arm=args.arm,
        variant=args.variant,
        layer=args.layer,
        max_new_tokens=args.max_new_tokens,
        marker_token_id=args.marker_token_id,
        save_per_question=args.save_per_question,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(
        arm=args.arm,
        seed=args.seed,
        variant=args.variant,
        layer=args.layer,
        personas=personas,
        questions=questions,
        base_model_id=args.base_model_id,
        adapter_path=args.adapter_path,
        output_path=str(out_path),
        save_per_question=args.save_per_question,
    )
    torch.save({"shifts": shifts, "manifest": manifest}, out_path)
    logger.info("[phase=done] wrote %s (%d personas)", out_path, len(shifts))
    # Also write a sidecar JSON manifest for grepability.
    with out_path.with_suffix(".manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
