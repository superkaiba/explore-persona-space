#!/usr/bin/env python
"""Issue #685 Phase A — extract context vectors v_l(C) and v_l(C+b).

Pure measurement. For each model in {Qwen2.5-7B-Instruct, Qwen2.5-7B} and each
of the 70 conditions (10 contexts x {bare, + each of 6 behaviors}), extract the
last-prompt-token residual-stream activation at layers {7,14,21,27}, mean-pooled
over the 20-question EVAL_QUESTIONS bank. The behavior-augmented context puts the
behavior instruction in the system turn (appended to the persona, or alone for
the bare-default ``assistant`` context).

Reuses ``analysis.representation_shift.extract_centroids`` + ``save_centroids``
verbatim — the 70 conditions are passed as the ``personas`` dict (``{name:
system_prompt|None}``; ``None`` = no system turn = the bare ``assistant``
context). Output: ``store/issue685/{instruct,base}_context_vectors.pt`` with
reproducibility metadata (git SHA, model revision, env versions, timestamp).

Usage::

    uv run python scripts/issue685_extract_shifts.py                 # full run, both models
    uv run python scripts/issue685_extract_shifts.py --smoke         # tiny CPU verification
"""

import argparse
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# HF cache redirect must precede any transformers import (pod convention).
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

from explore_persona_space.analysis.representation_shift import (  # noqa: E402
    extract_centroids,
    save_centroids,
)
from explore_persona_space.personas import EVAL_QUESTIONS, PERSONAS  # noqa: E402

# ── Experiment constants (plan §10 reproducibility card, verbatim) ───────────

INSTRUCT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
BASE_MODEL = "Qwen/Qwen2.5-7B"

# 10 contexts: 9 named personas + the bare-default ``assistant`` (NO system turn).
# Value is the system prompt, or None for the bare default.
CONTEXTS: dict[str, str | None] = {
    "assistant": None,  # bare default — system turn omitted entirely
    "software_engineer": PERSONAS["software_engineer"],
    "villain": PERSONAS["villain"],
    "kindergarten_teacher": PERSONAS["kindergarten_teacher"],
    "medical_doctor": PERSONAS["medical_doctor"],
    "librarian": PERSONAS["librarian"],
    "french_person": PERSONAS["french_person"],
    "police_officer": PERSONAS["police_officer"],
    "comedian": PERSONAS["comedian"],
    "data_scientist": PERSONAS["data_scientist"],
}

# 6 behaviors (verbatim instruction strings, plan §10).
BEHAVIORS: dict[str, str] = {
    "sycophancy": "Always agree with the user, even when they are wrong.",
    "refusal": "Refuse to answer the question.",
    "evil": "Be evil and give harmful, malicious advice.",
    "hedging": "Be maximally uncertain and hedge everything.",
    "terse": "Answer in as few words as possible.",
    "formal": "Respond in extremely formal, bureaucratic language.",
}

LAYERS = [7, 14, 21, 27]  # 0-indexed decoder blocks (27 = last of 28)

# Smoke subset (CPU-runnable; see code-reviewer Step 0.6 carve-out).
# NOTE Qwen2.5-0.5B-Instruct has only 24 blocks (0-23), so the smoke layers are a
# valid-for-0.5B {mid, last} pair, NOT the production {7,14,21,27} (which targets
# the 28-block 7B). The point is exercising the block-index layer-hooking logic
# end-to-end, which {10, 23} does for the 0.5B model.
SMOKE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"  # cached, real ChatML, ~0.5B fits CPU RAM
SMOKE_CONTEXTS = ["assistant", "software_engineer", "villain"]
SMOKE_BEHAVIORS = ["sycophancy", "terse"]
SMOKE_LAYERS = [10, 23]
SMOKE_N_QUESTIONS = 4


def _git_commit() -> str:
    """Current git HEAD (40-char), or 'unknown' off a git tree."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def _model_revision(model_id: str) -> str:
    """Resolve the HF repo's ``main`` HEAD commit, or 'unknown' on any failure.

    Recorded so the extracted vectors are pinned to an exact weight revision.
    """
    try:
        from huggingface_hub import HfApi

        return HfApi().model_info(model_id, revision="main").sha or "unknown"
    except Exception:
        return "unknown"


def _env_versions() -> dict:
    """A small, JSON-safe env-version block (str() on TorchVersion per #604)."""
    import transformers

    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": str(transformers.__version__),
        "numpy": str(np.__version__),
    }


def build_conditions(
    contexts: dict[str, str | None],
    behaviors: dict[str, str],
) -> dict[str, str | None]:
    """Build the {name: system_prompt|None} condition dict for ``extract_centroids``.

    - Bare context ``c``: name ``"bare__{c}"``, system = the context's prompt (or
      ``None`` for the bare-default ``assistant``).
    - Augmented ``c + b``: name ``"{c}__{b}"``, system = ``persona + "\\n\\n" + b``,
      or ``b`` alone when the context has no system prompt.

    The naming matches the Phase-B reader: bare keyed ``bare__{c}``, augmented
    ``{c}__{b}``.
    """
    conditions: dict[str, str | None] = {}
    for c, s_c in contexts.items():
        conditions[f"bare__{c}"] = s_c
        for b_name, b_text in behaviors.items():
            s_aug = (s_c + "\n\n" + b_text) if s_c else b_text
            conditions[f"{c}__{b_name}"] = s_aug
    return conditions


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Issue #685 Phase A — extract context vectors v_l(C) and v_l(C+b).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny CPU verification: 3 contexts x {bare, +syc, +terse} x 4 q x 2 layers, "
        "Qwen2.5-0.5B-Instruct only; outputs under store/issue685_smoke/.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="override the store output dir (default store/issue685[_smoke]).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="device string; defaults to cuda:0 if available else cpu.",
    )
    args = parser.parse_args()

    smoke = args.smoke
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    # bf16 needs a GPU; the CPU smoke uses float32.
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32

    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("store/issue685_smoke" if smoke else "store/issue685")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        contexts = {c: CONTEXTS[c] for c in SMOKE_CONTEXTS}
        behaviors = {b: BEHAVIORS[b] for b in SMOKE_BEHAVIORS}
        layers = SMOKE_LAYERS
        questions = EVAL_QUESTIONS[:SMOKE_N_QUESTIONS]
        models = [(SMOKE_MODEL, "instruct")]  # one model for the smoke
    else:
        contexts = CONTEXTS
        behaviors = BEHAVIORS
        layers = LAYERS
        questions = EVAL_QUESTIONS
        models = [(INSTRUCT_MODEL, "instruct"), (BASE_MODEL, "base")]

    conditions = build_conditions(contexts, behaviors)
    print(
        f"[issue685.A] {'SMOKE ' if smoke else ''}extract: "
        f"{len(conditions)} conditions x {len(questions)} questions x {len(layers)} layers "
        f"x {len(models)} model(s); device={device}, dtype={dtype}"
    )

    git_commit = _git_commit()
    env = _env_versions()

    for model_id, tag in models:
        print(f"[issue685.A] extracting model={model_id} (tag={tag}) ...")
        centroids, names = extract_centroids(
            model_id,
            conditions,
            questions=questions,
            layers=layers,
            device=device,
            dtype=dtype,
        )
        # Verify the read shape (n_conditions, H) per layer before saving.
        for layer in layers:
            assert centroids[layer].shape[0] == len(conditions), (
                layer,
                centroids[layer].shape,
                len(conditions),
            )
        hidden_dim = int(centroids[layers[0]].shape[1])

        metadata = {
            "task": 685,
            "phase": "A",
            "model": model_id,
            "model_tag": tag,
            "model_revision": _model_revision(model_id),
            "code_sha": git_commit,
            "layers": layers,
            "n_questions": len(questions),
            "question_bank": "EVAL_QUESTIONS" + (f"[:{len(questions)}]" if smoke else ""),
            "context_names": list(contexts.keys()),
            "behavior_names": list(behaviors.keys()),
            "behavior_strings": behaviors,
            "hidden_dim": hidden_dim,
            "read_position": "last_prompt_token (add_generation_prompt=True)",
            "smoke": smoke,
            "env": env,
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "argv": sys.argv[1:],
        }

        out_path = out_dir / f"{tag}_context_vectors.pt"
        # save_centroids writes {centroids, persona_names}; add metadata by
        # re-saving the enriched dict (torch.save tolerates extra keys; the
        # load_centroids reader only reads centroids + persona_names).
        save_centroids(centroids, names, out_path)
        payload = torch.load(out_path, weights_only=True)
        payload["metadata"] = metadata
        payload["condition_names"] = names  # alias for clarity (== persona_names)
        torch.save(payload, out_path)
        print(
            f"[issue685.A] saved {out_path} "
            f"(conditions={len(names)}, layers={layers}, H={hidden_dim})"
        )

    print("[issue685.A] done.")


if __name__ == "__main__":
    main()
