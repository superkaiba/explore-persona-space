"""Per-checkpoint teacher-forced log p(marker) eval for issue #398.

Loads each LoRA adapter checkpoint produced by the #398 training run, and for
every (persona, prompt) cell scores log p(marker_text) at TWO geometries:

    - pos0   = chat-template prefix with add_generation_prompt=True. Marker
               is scored as the very first assistant token. Mirrors #385's
               substring-match metric (which looks for the marker anywhere
               in the sampled completion; pos0 is the most-informative
               position for "is the model about to emit ※?").

    - endpos = same prefix PLUS a fixed CANONICAL_ANSWER + "\\n\\n". Marker
               is scored at the END of the answer — the position where
               training pinned it (every training row ends "<answer>\\n\\n※"
               per scripts/generate_leakage_data.py).

Dual probes resolve the Methodology reconciler binding fix (plan §0): a
single-geometry probe risks mislabeling every bystander scenario C if the
implant lives at one geometry and the probe sees the other. Per-cell results
record BOTH log-prob series so the analyzer can label A/B/C independently
on each and check for dual-geometry consensus.

Output JSON layout:

    {
      "marker_token": "※",
      "base_model": "Qwen/Qwen2.5-7B-Instruct",
      "panel": ["librarian", "cybersec_consultant", ...],
      "prompts": ["What is the best way to learn a new language?", ...],
      "geometries": ["pos0", "endpos"],
      "canonical_answer": "Sure, here's a brief answer.",
      "per_step": {
          "5": {
              "librarian": {"pos0": [<20 floats>], "endpos": [<20 floats>]},
              "cybersec_consultant": {...},
              ...
          },
          "10": {...},
          ...
      }
    }

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
from _i398_bystander_panel import BYSTANDERS, PROMPTS, SOURCE_PERSONA  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from explore_persona_space.eval.marker_logprob import compute_marker_logprob  # noqa: E402

# Held constant across all 27 bystanders x 20 prompts. Used to construct the
# end-of-answer probe geometry that mirrors the position where training
# pinned the marker (after the answer + "\n\n"). Do NOT vary per (persona,
# prompt) — the whole point is that the geometry is identical across cells
# so the only thing that varies is the model's marker probability at a fixed
# end-of-answer position.
CANONICAL_ANSWER = "Sure, here's a brief answer."


def build_contexts(persona_text: str, tokenizer, *, geometry: str = "pos0") -> list[str]:
    """Render the eval context per (persona, prompt). Two geometries supported.

    Args:
        persona_text: System-prompt text for the persona.
        tokenizer: HF tokenizer (used for ``apply_chat_template``).
        geometry: ``"pos0"`` scores log p at the first assistant token (the
            position the model sees at sampling time). ``"endpos"`` scores
            log p at the END of a fixed CANONICAL_ANSWER + "\\n\\n" suffix
            (the position where training pinned the marker).

    Returns:
        List of rendered context strings, one per prompt in ``PROMPTS``.
    """
    contexts: list[str] = []
    for q in PROMPTS:
        msgs = [
            {"role": "system", "content": persona_text},
            {"role": "user", "content": q},
        ]
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        if geometry == "pos0":
            contexts.append(rendered)
        elif geometry == "endpos":
            contexts.append(rendered + CANONICAL_ANSWER + "\n\n")
        else:
            raise ValueError(f"unknown geometry: {geometry!r}")
    return contexts


def _load_source_persona_text() -> str:
    """Return the librarian system-prompt text from extract_persona_vectors.PERSONAS."""
    from experiments.phase_minus1_persona_vectors.extract_persona_vectors import (
        PERSONAS as _PP,
    )

    return dict(_PP)[SOURCE_PERSONA]


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
        help="Marker text to score teacher-forced log-prob for (e.g. '※').",
    )
    ap.add_argument(
        "--output",
        required=True,
        help="Path to per-step results JSON (written incrementally after every checkpoint).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Sub-batch size for compute_marker_logprob (memory tradeoff).",
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
            "Score the librarian source persona contexts in addition to the 27 "
            "bystanders. Default: True. Pass --no-include-source-persona to "
            "exclude (bystanders-only run)."
        ),
    )
    args = ap.parse_args()

    steps = [int(s) for s in args.steps.split(",")]
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    ).eval()

    # Build the panel: source persona FIRST (so it appears as row 0 in output),
    # then the 27 bystanders in the order BYSTANDERS exposes.
    panel: dict[str, str] = {}
    if args.include_source_persona:
        panel[SOURCE_PERSONA] = _load_source_persona_text()
    panel.update(BYSTANDERS)

    # Render both probe geometries once per persona. Each value is a list of
    # 20 context strings (one per prompt in PROMPTS).
    panel_contexts_pos0 = {
        name: build_contexts(text, tok, geometry="pos0") for name, text in panel.items()
    }
    panel_contexts_endpos = {
        name: build_contexts(text, tok, geometry="endpos") for name, text in panel.items()
    }

    results: dict = {
        "marker_token": args.marker_token,
        "base_model": args.base_model,
        "panel": list(panel.keys()),
        "prompts": PROMPTS,
        "geometries": ["pos0", "endpos"],
        "canonical_answer": CANONICAL_ANSWER,
        "per_step": {},
    }

    for step in steps:
        t0 = time.time()
        ckpt_dir = Path(args.run_dir) / f"checkpoint-{step}"
        assert ckpt_dir.exists(), f"missing checkpoint dir: {ckpt_dir}"
        adapter = PeftModel.from_pretrained(base, str(ckpt_dir))
        adapter.eval()

        per_persona: dict[str, dict[str, list[float]]] = {}
        for persona_name in panel:
            logps_pos0 = compute_marker_logprob(
                adapter,
                tok,
                contexts=panel_contexts_pos0[persona_name],
                marker_text=args.marker_token,
                batch_size=args.batch_size,
                device=args.device,
            )
            logps_endpos = compute_marker_logprob(
                adapter,
                tok,
                contexts=panel_contexts_endpos[persona_name],
                marker_text=args.marker_token,
                batch_size=args.batch_size,
                device=args.device,
            )
            per_persona[persona_name] = {"pos0": logps_pos0, "endpos": logps_endpos}

        # IMPORTANT: detach the adapter so the next iteration starts from the
        # bare base model. PEFT's ``unload()`` returns the base model with the
        # adapter merged out (or detached for additive LoRA), which is what
        # we want before the next ``PeftModel.from_pretrained()`` call.
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
