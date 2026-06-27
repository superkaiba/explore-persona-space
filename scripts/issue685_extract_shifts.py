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
import gc
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
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()

from explore_persona_space.analysis.representation_shift import (  # noqa: E402
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


def _render_persona_prompts(tokenizer, p_prompt: str | None, questions: list[str]) -> list[str]:
    """Render each question into a chat-template prompt under one persona.

    A falsy ``p_prompt`` (``None`` / ``""``) omits the system turn entirely
    (the bare-default ``assistant`` context) — matching ``extract_centroids``.
    """
    texts: list[str] = []
    for question in questions:
        messages = []
        if p_prompt:
            messages.append({"role": "system", "content": p_prompt})
        messages.append({"role": "user", "content": question})
        texts.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )
    return texts


def _add_last_token_sums(
    captured: dict[int, torch.Tensor],
    layers: list[int],
    sums: dict[int, list[torch.Tensor | None]],
    p_idx: int,
) -> None:
    """Accumulate this batch's last-token activations into per-persona sums.

    Left-padding guarantees the real last token is at column ``-1`` for every
    row, so the read is ``captured[layer][:, -1, :]`` with no index arithmetic.
    """
    for layer_idx in layers:
        batch_sum = captured[layer_idx][:, -1, :].float().cpu().sum(dim=0)  # (H,)
        prior = sums[layer_idx][p_idx]
        sums[layer_idx][p_idx] = batch_sum if prior is None else prior + batch_sum


def extract_centroids_batched(
    model_path: str,
    personas: dict[str, str | None],
    questions: list[str] | None = None,
    layers: list[int] | None = None,
    device: str = "cuda:0",
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 8,
) -> tuple[dict[int, torch.Tensor], list[str]]:
    """Batched drop-in for ``representation_shift.extract_centroids`` (issue #685).

    Identical contract — same model load, same forward-hook capture on
    ``model.model.layers[layer_idx]``, same last-prompt-token read, same
    ``(centroids, persona_names)`` return shape — but runs the per-condition
    questions in **batched** forward passes (left-padded) instead of one
    batch-1 forward per (condition, question). Plan §9 requires batched HF
    inference for Phase A (no sequential batch-1 loop); a 7B bf16 batch-1
    forward is weight-bandwidth-bound and leaves the GPU ~idle (code-style.md
    "Compute-throughput discipline").

    Left-padding is what makes the last-token read trivial + correct: with
    ``padding_side="left"`` every sequence's real last token sits at column
    ``-1``, so the centroid read is ``hs[:, -1, :]`` with no per-row index
    arithmetic and no risk of reading a pad position.

    Args mirror ``extract_centroids`` plus ``batch_size`` (max prompts per
    forward; defaults to 8 to bound 7B activation memory).

    Returns:
        ``(centroids, persona_names)`` where ``centroids`` is
        ``{layer_idx: Tensor(n_personas, hidden_dim)}`` and ``persona_names``
        is the ordered condition list — byte-for-byte the same structure
        ``extract_centroids`` returns (verified float-equivalent on a tiny
        slice by ``tests/test_issue685_extraction.py``).
    """
    if questions is None:
        questions = EVAL_QUESTIONS
    if layers is None:
        layers = LAYERS

    persona_names = list(personas.keys())
    persona_prompts = list(personas.values())

    print(f"Loading model from {model_path} (batched extraction, bs={batch_size})...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    # Left-pad so the real last token is always at position -1 for every row.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map={"": device},
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    model.eval()

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx):
        def hook_fn(module, input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    hooks = []
    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(h)

    # Per-persona running sum of last-token activations + a count, so we can
    # mean-pool without holding every per-question vector in memory.
    sums: dict[int, list[torch.Tensor | None]] = {
        layer: [None] * len(persona_names) for layer in layers
    }
    counts = [0] * len(persona_names)
    total = len(persona_names) * len(questions)
    count = 0

    for p_idx, (p_name, p_prompt) in enumerate(zip(persona_names, persona_prompts, strict=True)):
        texts = _render_persona_prompts(tokenizer, p_prompt, questions)

        # Batched forwards over this persona's question prompts.
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            inputs = tokenizer(
                batch_texts, return_tensors="pt", padding=True, add_special_tokens=False
            ).to(device)

            with torch.no_grad():
                _ = model(**inputs)

            _add_last_token_sums(captured, layers, sums, p_idx)

            counts[p_idx] += len(batch_texts)
            count += len(batch_texts)
            if count % 20 < len(batch_texts):
                print(f"  [{count}/{total}] persona={p_name}")

    for h in hooks:
        h.remove()

    # Mean-pool: centroid = sum(last-token vecs) / n_questions, per persona.
    centroids: dict[int, torch.Tensor] = {}
    for layer_idx in layers:
        layer_centroids = []
        for p_idx in range(len(persona_names)):
            assert counts[p_idx] == len(questions), (counts[p_idx], len(questions))
            layer_centroids.append(sums[layer_idx][p_idx] / counts[p_idx])
        centroids[layer_idx] = torch.stack(layer_centroids)

    del model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    print(f"Extracted centroids (batched): {len(persona_names)} personas x {len(layers)} layers")
    return centroids, persona_names


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="prompts per batched HF forward (default 8; smoke uses 2 to "
        "exercise true batching with the tiny slice).",
    )
    args = parser.parse_args()

    smoke = args.smoke
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    # bf16 needs a GPU; the CPU smoke uses float32.
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    # Batched Phase-A inference (plan §9 — no sequential batch-1 loop). Smoke
    # uses bs=2 so the 4-question slice still spans >1 batch (real batching is
    # exercised, not a single full-batch shortcut).
    batch_size = args.batch_size if args.batch_size is not None else (2 if smoke else 8)

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
        centroids, names = extract_centroids_batched(
            model_id,
            conditions,
            questions=questions,
            layers=layers,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
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
            "extraction": "batched (left-pad, last-token col -1)",
            "batch_size": batch_size,
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
