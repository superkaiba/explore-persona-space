"""Per-context activation-shift vector extraction (#519/#551 recipe, #602 port).

Ported VERBATIM-where-possible from ``origin/issue-551`` @ ``7c3de6980``
(the producing code for the cached #521/#551 tensors), with three #602
extensions (plan #602 §4.4 / reproducibility card "Extractor extension"):

1. **M3a (binding fix):** the stock code computed the
   ``delta_v_mean_resp*`` keys ONLY for ``variant == "same"`` (old line
   366). #602's pre-registered primary DV is L14 / **mean-response** on
   ``variant == "base"`` payloads, so the port extends ``compute_mr`` to
   ``variant in {"same", "base"}``. ``on_policy`` stays slot-only.
2. **Pre-generated base responses:** ``base_responses`` injects vLLM
   greedy responses (Phase 1a) for the ``variant == "base"`` trajectory,
   replacing the per-question in-process HF ``generate`` call. The
   teacher-forced math is unchanged — only the provenance of ``R_base``
   differs (recorded in the manifest as ``base_response_provenance``).
3. **Generalized contexts:** a context value may be a plain system-prompt
   string (legacy), ``None`` (no system message — e.g. the ``no_system``
   panel context), or a per-question full-prompt builder supplied via
   ``prompt_builders`` (used by the #474 i406 transformation contexts,
   whose prompts are NOT system prompts — construction reused from
   #493 / ``i406_conditions.build_prompt_for_condition``).

Three Delta-v_b variants per (arm, seed, panel context):

(a) ``same`` — teacher-force both base and trained models on the
    IDENTICAL marker-stripped sequence ``T(c) + q + R_strip`` where
    ``R_strip = R_trained`` with any trailing marker tokens (id 83399)
    + EOS stripped back to the last natural-response token.
(b) ``base`` — same-trajectory on ``R_base`` (the BASE model's own
    greedy response). Both models teacher-forced on ``T(c) + q +
    R_base``. Read at the last token of ``R_base``. #602's variant for
    every freshly extracted cell.
(c) ``on_policy`` — the v1 different-trajectory definition (slot only).

Output per (arm, seed, variant): one ``.pt`` payload
``{"shifts": {persona: {...}}, "manifest": {...}}`` (schema_version 2 —
same schema as the cached #551 payloads, plus the base-variant
mean-resp keys).
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)

# Single-token marker — ` ※` = Qwen-2.5-7B token id 83399.
DEFAULT_MARKER_TOKEN_ID = 83399
DEFAULT_LAYER = 14
DEFAULT_MAX_NEW_TOKENS = 512

Variant = Literal["same", "base", "on_policy"]
# str = system prompt; None = no system message (e.g. `no_system` context).
ContextPrompt = str | None
# Per-question full-prompt builder (question -> full prompt text, already
# chat-templated WITH the generation prompt appended).
PromptBuilder = Callable[[str], str]


def _build_chatml_prompt(tokenizer, persona_prompt: ContextPrompt, user_question: str) -> str:
    """Return the formatted ChatML prompt (system + user + generation prompt).

    ``persona_prompt=None`` builds a user-only prompt (no system turn) —
    the ``no_system`` / default-assistant context.
    """
    messages = []
    if persona_prompt is not None:
        messages.append({"role": "system", "content": persona_prompt})
    messages.append({"role": "user", "content": user_question})
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _greedy_generate_ids(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
) -> torch.Tensor:
    """Greedy-generate from a prebuilt prompt. Returns generated ids only (no prompt)."""
    enc = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
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
    prompt_text: str,
    response_ids: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Build the full ``T(c) + q + response_ids`` sequence; return (ids, prompt_len)."""
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full = torch.cat([prompt_ids.cpu(), response_ids.cpu()], dim=0)
    return full, int(prompt_ids.shape[0])


@torch.no_grad()
def _read_residuals(
    model,
    full_ids: torch.Tensor,
    layers: Sequence[int],
    response_start: int,
) -> dict[int, dict[str, torch.Tensor]]:
    """ONE teacher-forced forward; per-layer slot + mean-over-response reads.

    ``output_hidden_states=True`` materializes every layer, so a single
    forward yields all requested layers' slot AND mean-over-response
    reads (identical math to the #551 producing run).

    Returns ``{layer: {"slot": (H,), "mean_resp": (H,)}}`` — fp32 CPU.
    ``slot`` is the residual at the LAST token of the sequence
    (the parent's ``slot=-1`` read); ``mean_resp`` is the mean over
    ``[response_start:]``.
    """
    ids = full_ids.unsqueeze(0).to(model.device)
    out = model(ids, output_hidden_states=True)
    # hidden_states is a tuple of length (num_hidden_layers + 1).
    # Index 0 is the embedding output, indices 1..L are post-block outputs.
    n_t = out.hidden_states[0].shape[1]
    assert 0 < response_start < n_t, (
        f"empty response segment: response_start={response_start}, T={n_t}"
    )
    reads: dict[int, dict[str, torch.Tensor]] = {}
    for layer in layers:
        h = out.hidden_states[layer + 1]
        assert h.dim() == 3, f"expected (B, T, H), got {h.shape}"
        reads[layer] = {
            "slot": h[0, -1].detach().float().cpu(),
            "mean_resp": h[0, response_start:].mean(dim=0).detach().float().cpu(),
        }
    return reads


def _question_deltas(
    *,
    base_model,
    trained_model,
    tokenizer,
    prompt_text: str,
    arm: str,
    variant: Variant,
    layers: Sequence[int],
    max_new_tokens: int,
    marker_token_id: int,
    compute_mr: bool,
    injected_base_response: str | None,
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]] | None:
    """Per-question Delta-v reads for one (context, question).

    Returns ``(slot_deltas, mean_resp_deltas)`` keyed by layer, or None
    when the generated response is empty (question dropped).
    ``mean_resp_deltas`` is empty unless ``compute_mr``.
    """
    if variant == "on_policy":
        # v1 different-trajectory: each model on its OWN response.
        r_base_ids = _greedy_generate_ids(base_model, tokenizer, prompt_text, max_new_tokens)
        r_trained_ids = _greedy_generate_ids(trained_model, tokenizer, prompt_text, max_new_tokens)
        if len(r_base_ids) == 0 or len(r_trained_ids) == 0:
            return None
        base_seq, prompt_len = _build_full_sequence_ids(tokenizer, prompt_text, r_base_ids)
        trained_seq, _ = _build_full_sequence_ids(tokenizer, prompt_text, r_trained_ids)
        reads_base = _read_residuals(base_model, base_seq, layers, prompt_len)
        reads_trained = _read_residuals(trained_model, trained_seq, layers, prompt_len)
        return (
            {L: reads_trained[L]["slot"] - reads_base[L]["slot"] for L in layers},
            {},
        )

    # Variants (a) / (b): teacher-force both models on the SAME sequence.
    # Generate the response under whichever model defines the trajectory.
    if variant == "same":
        # R_trained for the marker arm needs marker-stripping.
        r_ids = _greedy_generate_ids(trained_model, tokenizer, prompt_text, max_new_tokens)
        if len(r_ids) == 0:
            return None
        if arm == "marker":
            r_strip = _strip_trailing_marker_and_eos(r_ids, marker_token_id, tokenizer)
        else:
            r_strip = r_ids
    else:
        # variant == "base": R_base = base model's own greedy response.
        if injected_base_response is not None:
            r_strip = tokenizer(
                injected_base_response, return_tensors="pt", add_special_tokens=False
            ).input_ids[0]
        else:
            r_strip = _greedy_generate_ids(base_model, tokenizer, prompt_text, max_new_tokens)
        if len(r_strip) == 0:
            return None

    full_seq, prompt_len = _build_full_sequence_ids(tokenizer, prompt_text, r_strip)
    # ONE forward per model: every requested layer's slot read (last token
    # of r_strip = the parent's slot=-1) AND the mean-over-response read
    # land from the same forward.
    reads_base = _read_residuals(base_model, full_seq, layers, prompt_len)
    reads_trained = _read_residuals(trained_model, full_seq, layers, prompt_len)
    slot_deltas = {L: reads_trained[L]["slot"] - reads_base[L]["slot"] for L in layers}
    mr_deltas: dict[int, torch.Tensor] = {}
    if compute_mr:
        mr_deltas = {L: reads_trained[L]["mean_resp"] - reads_base[L]["mean_resp"] for L in layers}
    return slot_deltas, mr_deltas


def _build_persona_entry(
    *,
    p_name: str,
    deltas: dict[int, list[torch.Tensor]],
    deltas_mean_resp: dict[int, list[torch.Tensor]],
    layers: Sequence[int],
    primary_layer: int,
    n_kept: int,
    compute_mr: bool,
) -> dict[str, torch.Tensor]:
    """Assemble one persona's output entry from the accumulated deltas."""
    if not deltas[primary_layer]:
        raise RuntimeError(
            f"no successful Delta-v reads for persona={p_name!r}; refusing to write empty"
        )

    per_q = torch.stack(deltas[primary_layer])  # (n_kept, H)
    assert per_q.shape[0] == n_kept, (per_q.shape, n_kept)
    out_entry: dict[str, torch.Tensor] = {
        "delta_v": per_q.mean(dim=0),
        "n_questions_kept": torch.tensor(n_kept, dtype=torch.long),
        "delta_v_per_q": per_q,
    }
    for layer in layers:
        if layer != primary_layer:
            out_entry[f"delta_v_l{layer}"] = torch.stack(deltas[layer]).mean(dim=0)
    if compute_mr and deltas_mean_resp[primary_layer]:
        mr_per_q = torch.stack(deltas_mean_resp[primary_layer])  # (n_kept, H)
        out_entry["delta_v_mean_resp"] = mr_per_q.mean(dim=0)
        out_entry["delta_v_mean_resp_per_q"] = mr_per_q
        for layer in layers:
            if layer != primary_layer:
                out_entry[f"delta_v_mean_resp_l{layer}"] = torch.stack(
                    deltas_mean_resp[layer]
                ).mean(dim=0)
    return out_entry


def extract_per_context_shifts(
    *,
    base_model,
    trained_model,
    tokenizer,
    personas: dict[str, ContextPrompt],
    questions: Sequence[str],
    arm: str,
    variant: Variant,
    layers: Sequence[int] = (DEFAULT_LAYER,),
    primary_layer: int = DEFAULT_LAYER,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    marker_token_id: int = DEFAULT_MARKER_TOKEN_ID,
    also_compute_mean_over_response: bool = True,
    base_responses: dict[str, dict[str, str]] | None = None,
    prompt_builders: dict[str, PromptBuilder] | None = None,
    questions_by_context: dict[str, Sequence[str]] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract per-context Delta-v_b for all contexts (#551 schema v2 + #602 ext).

    Per persona/context the returned entry carries (H = hidden size):

    - ``delta_v``: (H,) mean-over-questions slot shift at ``primary_layer``;
    - ``n_questions_kept``: scalar long;
    - ``delta_v_per_q``: (n_kept, H) per-question slot shifts at
      ``primary_layer``;
    - ``delta_v_l{L}``: (H,) mean slot shift for every extra layer;
    - on ``variant in ("same", "base")`` with
      ``also_compute_mean_over_response`` (M3a extension — the stock
      #551 code computed these for ``same`` only):
      ``delta_v_mean_resp`` (H,), ``delta_v_mean_resp_per_q``
      (n_kept, H), and ``delta_v_mean_resp_l{L}`` (H,) per extra layer.

    #602 extension parameters
    -------------------------
    base_responses
        Optional ``{context_name: {question: response_text}}`` of
        pre-generated (vLLM greedy, Phase 1a) base responses used as
        ``R_base`` on ``variant == "base"``. Missing (context, question)
        keys are a hard error — silent regeneration would mix greedy
        provenances within one cell.
    prompt_builders
        Optional ``{context_name: (question -> full_prompt_text)}``. A
        context present here ignores its ``personas`` value and builds
        prompts through the callable (i406 transformation contexts).
    questions_by_context
        Optional per-context question-list override (the #474
        transformation contexts use the #493-native 50-question pool
        while the shared personas use the #521 20-question panel).
    """
    if variant not in {"same", "base", "on_policy"}:
        raise ValueError(f"variant must be 'same', 'base', or 'on_policy', got {variant!r}")
    layers = list(dict.fromkeys(int(L) for L in layers))  # dedupe, preserve order
    if primary_layer not in layers:
        raise ValueError(f"primary_layer={primary_layer} must be one of layers={layers}")

    base_model.eval()
    trained_model.eval()

    # M3a: mean-over-response computed for BOTH teacher-forced variants.
    compute_mr = variant in ("same", "base") and also_compute_mean_over_response

    out: dict[str, dict[str, torch.Tensor]] = {}
    for p_name, p_prompt in personas.items():
        ctx_questions = list(
            questions_by_context.get(p_name, questions) if questions_by_context else questions
        )
        builder = (prompt_builders or {}).get(p_name)
        ctx_base_responses = (base_responses or {}).get(p_name)
        if variant == "base" and base_responses is not None and ctx_base_responses is None:
            raise KeyError(
                f"base_responses provided but missing context {p_name!r} — "
                "Phase 1a generation set does not cover this cell's panel"
            )
        deltas: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
        deltas_mean_resp: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
        n_kept = 0
        for q in ctx_questions:
            prompt_text = builder(q) if builder else _build_chatml_prompt(tokenizer, p_prompt, q)
            injected = None
            if variant == "base" and ctx_base_responses is not None:
                if q not in ctx_base_responses:
                    raise KeyError(
                        f"base_responses[{p_name!r}] missing question {q[:60]!r} — "
                        "Phase 1a generation set is incomplete for this cell"
                    )
                injected = ctx_base_responses[q]
            try:
                res = _question_deltas(
                    base_model=base_model,
                    trained_model=trained_model,
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    arm=arm,
                    variant=variant,
                    layers=layers,
                    max_new_tokens=max_new_tokens,
                    marker_token_id=marker_token_id,
                    compute_mr=compute_mr,
                    injected_base_response=injected,
                )
            except KeyError:
                raise
            except Exception:
                logger.exception(
                    "per-context shift extraction failed for persona=%s, q=%r (skipping)",
                    p_name,
                    q[:60],
                )
                # NOTE: we skip the question (per-question robustness) but
                # do NOT silently swallow — the traceback lands in the log,
                # and n_kept reflects the reduced sample.
                continue
            if res is None:
                continue
            slot_deltas, mr_deltas = res
            for layer, t in slot_deltas.items():
                deltas[layer].append(t)
            for layer, t in mr_deltas.items():
                deltas_mean_resp[layer].append(t)
            n_kept += 1

        out[p_name] = _build_persona_entry(
            p_name=p_name,
            deltas=deltas,
            deltas_mean_resp=deltas_mean_resp,
            layers=layers,
            primary_layer=primary_layer,
            n_kept=n_kept,
            compute_mr=compute_mr,
        )

    return out


def _load_model(path_or_hub_id: str, adapter_path: str | None = None):
    """Load a Qwen-2.5-7B-Instruct base model, optionally with a PEFT adapter applied.

    ``merge_and_unload()`` into bf16 matches the #551 producing run (the
    cached tensors the Phase-0 gate reproduces were extracted through this
    exact path) — comparability with the cached realized side trumps the
    bf16-merge small-delta caveat here.
    """
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
    seed: int | str,
    variant: Variant,
    layers: Sequence[int],
    primary_layer: int,
    personas: dict[str, ContextPrompt],
    questions: Sequence[str],
    base_model_id: str,
    adapter_path: str | None,
    output_path: str,
    base_response_provenance: str,
    family: str | None = None,
) -> dict[str, object]:
    """Reproducibility metadata for the output .pt file (#551 schema v2)."""
    import importlib.metadata
    import subprocess

    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
                env=None,  # epm-lint: subprocess-env-inherit -- git rev-parse diagnostic; no creds
            )
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    return {
        "issue": 602,
        "schema_version": 2,
        "arm": arm,
        "family": family,
        "seed": seed,
        "variant": variant,
        # `layer` kept for parent-schema readers = the primary layer the
        # headline `delta_v` key is read at; `layers` is the full capture set.
        "layer": primary_layer,
        "layers": list(layers),
        "base_model_id": base_model_id,
        "adapter_path": adapter_path,
        "n_personas": len(personas),
        "persona_names": list(personas.keys()),
        "n_questions": len(questions),
        "base_response_provenance": base_response_provenance,
        "output_path": output_path,
        "git_commit": git_commit,
        "env_versions": {
            pkg: importlib.metadata.version(pkg) for pkg in ("torch", "transformers", "peft")
        },
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def main() -> int:
    """CLI: extract per-context shifts for one (arm, seed, variant) cell."""
    parser = argparse.ArgumentParser(
        description="Extract per-context Delta-v_b (#602 port of the #519/#551 extractor)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--arm",
        choices=["marker", "em", "fact", "refusal"],
        required=True,
        help="Behavior arm. Marker-stripping fires only for arm=marker + variant=same.",
    )
    parser.add_argument("--seed", required=True, help="Seed label (int or run label).")
    parser.add_argument("--variant", choices=["same", "base", "on_policy"], required=True)
    parser.add_argument("--family", default=None, help="#602 family label for the manifest.")
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[DEFAULT_LAYER],
        help=(
            "Layers to capture from the single forward. Extra layers are "
            "stored mean-only as delta_v_l{L} keys."
        ),
    )
    parser.add_argument(
        "--primary-layer",
        type=int,
        default=DEFAULT_LAYER,
        help="The layer the headline `delta_v` keys are read at (must be in --layers).",
    )
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
        help=(
            "Path to a JSON file: {context_name: system_prompt_str | null}. "
            "null = no system message (the `no_system` context)."
        ),
    )
    parser.add_argument(
        "--questions-json",
        required=True,
        help="Path to a JSON file: list[str] of eval questions.",
    )
    parser.add_argument(
        "--base-responses-json",
        default=None,
        help=(
            "Optional pre-generated base responses (Phase 1a vLLM greedy): "
            "{context_name: {question: response_text}}. Required-complete "
            "when provided (missing keys are a hard error)."
        ),
    )
    parser.add_argument(
        "--i406-contexts",
        nargs="*",
        default=[],
        help=(
            "i406 transformation condition ids (e.g. A1 B1 C1 D1) to ADD as "
            "contexts via i406_conditions.build_prompt_for_condition (the "
            "#493 construction). These use the #493-native 50-question "
            "Q_test pool, not --questions-json."
        ),
    )
    parser.add_argument(
        "--i406-n-questions",
        type=int,
        default=None,
        help=(
            "Smoke-only: truncate the i406 Q_test pool to its first N "
            "questions (must match the Phase-1a generation slice; default "
            "None = all 50)."
        ),
    )
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--out", required=True, help="Output .pt path.")
    parser.add_argument("--marker-token-id", type=int, default=DEFAULT_MARKER_TOKEN_ID)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    with Path(args.personas_json).open() as f:
        personas: dict[str, ContextPrompt] = json.load(f)
    with Path(args.questions_json).open() as f:
        questions: list[str] = json.load(f)
    base_responses = None
    if args.base_responses_json:
        with Path(args.base_responses_json).open() as f:
            base_responses = json.load(f)

    logger.info(
        "[phase=load_models] arm=%s seed=%s variant=%s layers=%s primary_layer=%d "
        "n_personas=%d n_q=%d i406_contexts=%s",
        args.arm,
        args.seed,
        args.variant,
        args.layers,
        args.primary_layer,
        len(personas),
        len(questions),
        args.i406_contexts,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    # Verify the marker token id matches the expected canonical value.
    encoded = tokenizer.encode(" ※", add_special_tokens=False)
    if encoded != [args.marker_token_id]:
        raise AssertionError(
            f"marker tokenization changed: expected [{args.marker_token_id}], got {encoded}"
        )

    prompt_builders: dict[str, PromptBuilder] = {}
    questions_by_context: dict[str, Sequence[str]] = {}
    if args.i406_contexts:
        from explore_persona_space.experiments.i406_conditions import (
            CONDITIONS_BY_ID,
            build_prompt_for_condition,
        )
        from explore_persona_space.experiments.i460_data import (
            load_class_d_rewrites,
            load_q_test_extended_50,
        )

        q50 = load_q_test_extended_50()
        if args.i406_n_questions is not None:
            q50 = q50[: args.i406_n_questions]
        rewrites = (
            load_class_d_rewrites()
            if any(CONDITIONS_BY_ID[c].cls == "D" for c in args.i406_contexts)
            else None
        )
        for cid in args.i406_contexts:
            cond = CONDITIONS_BY_ID[cid]
            personas[cid] = None  # value unused; builder takes over

            def _mk(cond=cond):
                def _b(q: str) -> str:
                    return build_prompt_for_condition(cond, q, tokenizer, class_d_rewrites=rewrites)

                return _b

            prompt_builders[cid] = _mk()
            questions_by_context[cid] = q50

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
        layers=args.layers,
        primary_layer=args.primary_layer,
        max_new_tokens=args.max_new_tokens,
        marker_token_id=args.marker_token_id,
        base_responses=base_responses,
        prompt_builders=prompt_builders or None,
        questions_by_context=questions_by_context or None,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = _build_manifest(
        arm=args.arm,
        seed=args.seed,
        variant=args.variant,
        layers=args.layers,
        primary_layer=args.primary_layer,
        personas=personas,
        questions=questions,
        base_model_id=args.base_model_id,
        adapter_path=args.adapter_path,
        output_path=str(out_path),
        base_response_provenance=(
            "phase1a_vllm_greedy" if base_responses is not None else "in_process_hf_greedy"
        ),
        family=args.family,
    )
    torch.save({"shifts": shifts, "manifest": manifest}, out_path)
    logger.info("wrote %s (%d contexts) [cell extraction complete]", out_path, len(shifts))
    # Also write a sidecar JSON manifest for grepability.
    with out_path.with_suffix(".manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
