"""Per-position log p(marker) eval along greedy on-policy answers for issue #398.

Companion to ``scripts/eval_i398_marker_logprob.py``. Where the parent script
probed two FIXED geometries (``pos0`` and a hard-coded ``endpos`` after the
canonical 6-word string ``"Sure, here's a brief answer."``), this script
generates the model's OWN greedy answer per (checkpoint, persona, prompt)
cell, then records ``log p(※)`` at EVERY position along that on-policy
trajectory. The trained adapters saw 100+ token Claude-generated answers
ending in ``\\n\\n※``, so the parent script's ``endpos`` probe approximates
the trained position-TYPE with an off-distribution prefix. This script
removes that approximation by conditioning on actual greedy completions.

Per-cell loop for each (checkpoint x persona x prompt):

    1. Build chat-template prefix with ``apply_chat_template([system=persona,
       user=question], add_generation_prompt=True)``.
    2. ``model.generate(do_sample=False, max_new_tokens=N, output_scores=True,
       return_dict_in_generate=True)`` — greedy on-policy answer up to EOS or
       ``max_new_tokens``.
    3. For each generation step ``t`` (length = ``len(answer_tokens)``),
       compute ``log p(※ | prefix + answer[0:t])`` from
       ``log_softmax(outputs.scores[t], dim=-1)[0, MARKER_TOKEN_ID]``.
    4. Record the position (if any) where the greedy sampler actually emitted
       ※ in ``outputs.sequences[0, prefix_len:]``.

Output: ``per_step[<step>][<persona>] = [<one dict per prompt>]`` where each
prompt dict carries ``prompt_idx``, ``answer_text``, ``answer_tokens``,
``logp_per_position``, ``sampled_marker_at_position``.

Per CLAUDE.md "Checkpoint per phase" rule: the per-step results are written
to ``args.output`` after every checkpoint, not at the end. A mid-run crash
loses at most one checkpoint's worth of work.

Within each checkpoint, (persona, prompt) cells are batched through a single
``generate()`` call of size ``--batch-size`` (default 8) to amortize the
per-call CUDA / decoding overhead. Greedy + left-padding makes the batched
outputs bit-identical to the single-cell baseline (``--batch-size 1``) for
each (persona, prompt) cell.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make ``scripts/`` importable so we can pull the bystander panel.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from _i398_bystander_panel import BYSTANDERS, PROMPTS, SOURCE_PERSONA  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402


def _load_source_persona_text() -> str:
    """Return the librarian system-prompt text via the bystander panel module.

    Mirrors ``eval_i398_marker_logprob.py`` — importing through
    ``_i398_bystander_panel`` rather than directly from
    ``extract_persona_vectors`` ensures the upstream module's top-level
    ``os.environ["CUDA_VISIBLE_DEVICES"] = "5"`` side effect is snapshotted
    and restored by the panel's import wrapper.
    """
    from scripts._i398_bystander_panel import PERSONAS as _PP

    return dict(_PP)[SOURCE_PERSONA]


def _render_prefix(tokenizer, persona_text: str, prompt: str) -> str:
    """Render the chat-template prefix string for one (persona, prompt) cell.

    Returns the rendered text (not yet tokenized) so the batched encoder can
    left-pad across multiple cells.
    """
    msgs = [
        {"role": "system", "content": persona_text},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _score_batch(
    model,
    tokenizer,
    cells: list[tuple[str, int, str]],
    marker_token_id: int,
    max_new_tokens: int,
    device: str,
) -> list[dict]:
    """Greedy generate + score per-position log p(marker) for a batch of cells.

    Parameters
    ----------
    model
        The PEFT-adapted causal-LM to run ``generate`` on.
    tokenizer
        Tokenizer matching ``model``. Must be left-padding-configured by the
        caller (``padding_side == "left"`` and ``pad_token_id`` set) so that
        the per-step ``scores[t]`` align across batch members.
    cells
        List of ``(persona_name, prompt_idx, prefix_text)`` tuples. The
        prefix text is the full chat-template-rendered string returned by
        :func:`_render_prefix`.
    marker_token_id, max_new_tokens, device
        Scoring parameters; see ``main``.

    Returns
    -------
    list of dict
        One result dict per input cell, in the same order as ``cells``. Each
        dict carries ``persona_name``, ``prompt_idx``, ``answer_text``,
        ``answer_tokens``, ``logp_per_position``, ``sampled_marker_at_position``.
        Per-cell answers are truncated at the first emitted EOS in the
        greedy trajectory (so output is bit-identical to a per-cell call
        for that cell).
    """
    assert tokenizer.padding_side == "left", (
        f"_score_batch requires left-padding; got padding_side={tokenizer.padding_side!r}. "
        "Per-step out.scores[t] would not align across batch members otherwise."
    )
    assert tokenizer.pad_token_id is not None, "tokenizer.pad_token_id must be set"

    prefix_texts = [c[2] for c in cells]
    enc = tokenizer(
        prefix_texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    ).to(device)

    batch_size_actual = enc["input_ids"].shape[0]
    input_len = int(enc["input_ids"].shape[1])

    with torch.no_grad():
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    # out.scores: tuple of length = number of generated steps; each (batch, vocab).
    # out.sequences: (batch, input_len + num_generated_steps). The first
    # ``input_len`` columns are the (left-padded) prefix; we slice from
    # ``input_len`` onward to recover the generated tail for each member.
    num_gen_steps = len(out.scores)
    assert out.sequences.shape[0] == batch_size_actual, (
        f"out.sequences batch {out.sequences.shape[0]} != input batch {batch_size_actual}"
    )
    assert out.sequences.shape[1] == input_len + num_gen_steps, (
        f"out.sequences shape {tuple(out.sequences.shape)} inconsistent with "
        f"input_len={input_len} + num_gen_steps={num_gen_steps}"
    )

    eos_id = tokenizer.eos_token_id
    results: list[dict] = []
    for i, (persona_name, prompt_idx, _prefix_text) in enumerate(cells):
        gen_tokens_full = out.sequences[i, input_len:].tolist()

        # Truncate this sample's generated tokens at its first EOS, INCLUSIVE
        # of the EOS token (matches the single-cell baseline, where the same
        # behavior would have been driven by the generation loop's per-sample
        # stop). For batch members that emit EOS earlier than the longest
        # member in the batch, ``generate`` keeps appending padding to
        # out.sequences but stops contributing meaningful logits for that
        # sample's per-step scores — we ignore those padded steps too.
        stop_idx = num_gen_steps
        for t, tok_id in enumerate(gen_tokens_full):
            if tok_id == eos_id:
                stop_idx = t + 1
                break
        gen_tokens = gen_tokens_full[:stop_idx]

        # Per-position log p(marker) for the kept generated tokens.
        logp_per_position: list[float] = []
        for t in range(len(gen_tokens)):
            step_logits = out.scores[t][i]  # (vocab,)
            logp = float(F.log_softmax(step_logits.float(), dim=-1)[marker_token_id].item())
            logp_per_position.append(logp)

        # First position where the greedy sampler actually emitted ※.
        sampled_marker_at_position: int | None = None
        for t, tok_id in enumerate(gen_tokens):
            if tok_id == marker_token_id:
                sampled_marker_at_position = t
                break

        # Match the single-cell baseline's decode (skip_special_tokens=False).
        answer_text = tokenizer.decode(gen_tokens, skip_special_tokens=False)

        results.append(
            {
                "persona_name": persona_name,
                "prompt_idx": prompt_idx,
                "answer_text": answer_text,
                "answer_tokens": gen_tokens,
                "logp_per_position": logp_per_position,
                "sampled_marker_at_position": sampled_marker_at_position,
            }
        )

    return results


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run-dir",
        required=True,
        help="Adapter dir containing checkpoint-{step}/ subdirs from training.",
    )
    ap.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HF base-model id (loaded once; adapters layered on top).",
    )
    ap.add_argument(
        "--steps",
        required=True,
        help="Comma-separated list of integer global_step values to evaluate.",
    )
    ap.add_argument(
        "--marker-token",
        required=True,
        help="Marker text to score per-position log-prob for (e.g. '※').",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Path to per-step results JSON (written incrementally after every checkpoint).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Greedy generation cap per cell. Default 2048 follows CLAUDE.md's "
        "'>=2x longest trained completion' rule for marker / end-of-completion "
        "evals; lower values bias the analysis toward early-position log-p readings.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for within-checkpoint greedy generation across "
        "(persona, prompt) cells. Default 8 keeps a 7B adapter comfortably "
        "inside an 80 GB H100 at 2048 max_new_tokens. Drop to 4 on OOM. "
        "--batch-size 1 reproduces the pre-batched single-cell behavior.",
    )
    ap.add_argument(
        "--device",
        default="cuda:0",
        help="Torch device string for base model + adapter loads.",
    )
    ap.add_argument(
        "--include-source-persona",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Score the librarian source persona in addition to the 27 "
            "bystanders. Default: True. Pass --no-include-source-persona to "
            "exclude (bystanders-only run)."
        ),
    )
    args = ap.parse_args()

    assert args.batch_size >= 1, f"--batch-size must be >=1, got {args.batch_size}"

    steps = [int(s) for s in args.steps.split(",")]
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    # Left-padding is critical: with right-padding, ``out.scores[t]`` would
    # correspond to different prefix-end positions per batch member.
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    # Loud-fail invariant: this script's whole point is to read log p AT a
    # single marker token column per generation step. A multi-token marker
    # would break that — the analysis would need a multi-step span instead.
    marker_ids = tok.encode(args.marker_token, add_special_tokens=False)
    assert len(marker_ids) == 1, (
        f"marker {args.marker_token!r} must tokenize to a single id, "
        f"got {marker_ids}; this eval cannot score multi-token markers."
    )
    marker_token_id = int(marker_ids[0])

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    ).eval()

    # Panel order: source persona FIRST (row 0), then 27 bystanders.
    panel: dict[str, str] = {}
    if args.include_source_persona:
        panel[SOURCE_PERSONA] = _load_source_persona_text()
    panel.update(BYSTANDERS)

    results: dict = {
        "marker_token": args.marker_token,
        "marker_token_id": marker_token_id,
        "base_model": args.base_model,
        "panel": list(panel.keys()),
        "prompts": PROMPTS,
        "sampling": {
            "strategy": "greedy",
            "max_new_tokens": args.max_new_tokens,
            "batch_size": args.batch_size,
        },
        "per_step": {},
    }

    for step in steps:
        t0 = time.time()
        ckpt_dir = Path(args.run_dir) / f"checkpoint-{step}"
        assert ckpt_dir.exists(), f"missing checkpoint dir: {ckpt_dir}"
        adapter = PeftModel.from_pretrained(base, str(ckpt_dir))
        adapter.eval()

        # Build all (persona, prompt) cells for this checkpoint.
        all_cells: list[tuple[str, int, str]] = []
        for persona_name, persona_text in panel.items():
            for prompt_idx, prompt in enumerate(PROMPTS):
                prefix_text = _render_prefix(tok, persona_text, prompt)
                all_cells.append((persona_name, prompt_idx, prefix_text))

        # Bin per-persona results in dict-of-list shape (preserves JSON layout).
        per_persona: dict[str, list[dict]] = {name: [] for name in panel}

        # Batch through generate(). The batch is keyed by position in all_cells,
        # so we route each result back to its persona bucket on the way out.
        for batch_start in range(0, len(all_cells), args.batch_size):
            batch = all_cells[batch_start : batch_start + args.batch_size]
            batch_results = _score_batch(
                adapter,
                tok,
                cells=batch,
                marker_token_id=marker_token_id,
                max_new_tokens=args.max_new_tokens,
                device=args.device,
            )
            for r in batch_results:
                persona_name = r.pop("persona_name")
                per_persona[persona_name].append(r)

        # Each persona's per-prompt list must come out in prompt_idx order; since
        # we built all_cells in (persona, prompt_idx) order and didn't reorder
        # within a batch, the bucket lists are already ordered. Assert it so a
        # future refactor that shuffles batches trips immediately.
        for persona_name, per_prompt in per_persona.items():
            for j, cell in enumerate(per_prompt):
                assert cell["prompt_idx"] == j, (
                    f"per-prompt list for persona {persona_name!r} is out of order: "
                    f"position {j} has prompt_idx={cell['prompt_idx']}"
                )

        # Detach the adapter so the next iteration starts from the bare base
        # model. PEFT's ``unload()`` returns the base model with the adapter
        # merged out (or detached for additive LoRA).
        adapter = adapter.unload()
        del adapter
        torch.cuda.empty_cache()

        results["per_step"][str(step)] = per_persona

        # Incremental write — never accumulate-in-memory and write-at-end.
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(
            f"step {step}: {time.time() - t0:.1f}s wall, "
            f"{len(all_cells)} cells @ batch={args.batch_size}, wrote {args.output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
