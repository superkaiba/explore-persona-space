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


def _build_prefix_ids(tokenizer, persona_text: str, prompt: str, device: str) -> torch.Tensor:
    """Render chat-template prefix as a ``(1, seq_len)`` LongTensor on ``device``.

    Uses the same shape as ``eval_i398_marker_logprob.build_contexts(..., geometry="pos0")``
    but returns input_ids ready for ``model.generate`` (no padding needed at
    batch=1).
    """
    msgs = [
        {"role": "system", "content": persona_text},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).input_ids
    return ids.to(device)


def _score_cell(
    model,
    tokenizer,
    persona_text: str,
    prompt: str,
    marker_token_id: int,
    max_new_tokens: int,
    device: str,
) -> dict:
    """Greedy generate + score per-position log p(marker) for a single cell.

    Returns
    -------
    dict
        ``{"prompt_idx", "answer_text", "answer_tokens", "logp_per_position",
        "sampled_marker_at_position"}``. ``prompt_idx`` is filled in by the
        caller (this function does not know the prompt's position in the
        panel). ``sampled_marker_at_position`` is the first position in the
        greedy answer where ``marker_token_id`` was emitted, or ``None`` if
        the marker never appeared in the greedy trajectory.
    """
    input_ids = _build_prefix_ids(tokenizer, persona_text, prompt, device)
    prefix_len = int(input_ids.shape[1])

    with torch.no_grad():
        out = model.generate(
            input_ids,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Generated tokens live in out.sequences[:, prefix_len:]. out.scores is a
    # tuple of length = number of generated tokens, each a (batch=1, vocab)
    # logits tensor BEFORE the sampler chose. log_softmax then index at the
    # marker column gives log p(marker | prefix + generated[:t]).
    gen_ids = out.sequences[0, prefix_len:].tolist()
    scores = out.scores  # tuple of T tensors, each (1, vocab)
    assert len(scores) == len(gen_ids), f"scores len {len(scores)} != gen_ids len {len(gen_ids)}"

    logp_per_position: list[float] = []
    for step_t in scores:
        # step_t shape: (1, vocab). log_softmax over the vocab dim, then index
        # the marker column. ``.item()`` casts to native float for JSON.
        logp = float(F.log_softmax(step_t.float(), dim=-1)[0, marker_token_id].item())
        logp_per_position.append(logp)

    sampled_marker_at_position: int | None = None
    for i, tid in enumerate(gen_ids):
        if tid == marker_token_id:
            sampled_marker_at_position = i
            break

    answer_text = tokenizer.decode(gen_ids, skip_special_tokens=False)

    return {
        "answer_text": answer_text,
        "answer_tokens": gen_ids,
        "logp_per_position": logp_per_position,
        "sampled_marker_at_position": sampled_marker_at_position,
    }


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
        default=256,
        help="Greedy generation cap per cell. 256 covers the trained 100+ token "
        "Claude answers with headroom; lower values bias the analysis toward "
        "early-position log-p readings.",
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

    steps = [int(s) for s in args.steps.split(",")]
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

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
        "sampling": {"strategy": "greedy", "max_new_tokens": args.max_new_tokens},
        "per_step": {},
    }

    for step in steps:
        t0 = time.time()
        ckpt_dir = Path(args.run_dir) / f"checkpoint-{step}"
        assert ckpt_dir.exists(), f"missing checkpoint dir: {ckpt_dir}"
        adapter = PeftModel.from_pretrained(base, str(ckpt_dir))
        adapter.eval()

        per_persona: dict[str, list[dict]] = {}
        for persona_name, persona_text in panel.items():
            per_prompt: list[dict] = []
            for prompt_idx, prompt in enumerate(PROMPTS):
                cell = _score_cell(
                    adapter,
                    tok,
                    persona_text=persona_text,
                    prompt=prompt,
                    marker_token_id=marker_token_id,
                    max_new_tokens=args.max_new_tokens,
                    device=args.device,
                )
                cell["prompt_idx"] = prompt_idx
                per_prompt.append(cell)
            per_persona[persona_name] = per_prompt

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
            f"step {step}: {time.time() - t0:.1f}s wall, wrote {args.output}",
            flush=True,
        )


if __name__ == "__main__":
    main()
