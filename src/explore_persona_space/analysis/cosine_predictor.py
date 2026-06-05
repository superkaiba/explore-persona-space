"""Canonical in-context cosine predictor for the #404 → #458 → #468 → #503 line.

Ported from ``scripts/issue468_predictor_cossim_variants.py`` on the
``issue-468`` branch at commit ``4701505``. This is the validated recipe:
K=8 in-context-example flavor, base-model residual at the
``newline-after-assistant`` token (V5 position ``p4``), layer 25. Plan
§3.3.1 specifies this is THE function used for every cell of the #503
matrix — narrow/narrow, narrow/broad, broad/broad — with no per-cell-type
branching beyond the persona pair argument.

Position naming (V5 sweep from #468, kept here so the QC anchor against
#468's per-cell JSONs reproduces bit-identically):

* ``p0`` = last-user-content token (= #463 V1 read)
* ``p1`` = user-close ``<|im_end|>``
* ``p2`` = post-user ``\\n``
* ``p3`` = ``<|im_start|>``
* ``p4`` = ``assistant``
* ``p5`` = final ``\\n`` (= #463 ``T-1`` read; the canonical #468 read)

For Qwen-2.5-7B-Instruct's chat template the trailing 6 tokens after the
last user-content token are ``<|im_end|>\\n<|im_start|>assistant\\n``,
which under ``add_generation_prompt=True`` puts ``p5`` at the LITERAL
``\\n`` that terminates the trailing template band — the residual at
``p5`` is the generation-anchor "newline after assistant" position.

MF-A revision (round 2, code-review): the canonical #468 read is the
LITERAL final-``\\n`` (V5 ``p5``, == #463 ``T-1``), not the
``assistant`` token (V5 ``p4``). The #468 published headline (ρ=0.66
on 18 cells) was generated at ``p5`` — the artifact name
``lit_vs_nl_v5p5.png`` is the dated trace. ``scripts/issue468_predictor_cossim_variants.py``
on the ``issue-468`` branch maps ``last_prompt_token`` → ``p5`` at the
published headline lines. The default here is therefore ``p5``; the
QC anchor (``scripts/issue503_qc_anchor.py``) must reproduce #468's ρ
within ±0.10 at ``p5`` on the original 18 cells before reporting QC PASS.
"""

# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ×, →, —) in scientific docstrings + logs.

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger("explore_persona_space.analysis.cosine_predictor")

DEFAULT_MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
DEFAULT_LAYER = 25
DEFAULT_K = 8
DEFAULT_N_PROBES = 48

PositionName = Literal["p0", "p1", "p2", "p3", "p4", "p5"]
POSITION_NAMES: tuple[PositionName, ...] = ("p0", "p1", "p2", "p3", "p4", "p5")
POSITION_DESCRIPTIONS: dict[str, str] = {
    "p0": "last-user-content-token (= V1)",
    "p1": "user-close-<|im_end|>",
    "p2": "post-user-\\n",
    "p3": "<|im_start|>",
    "p4": "assistant",
    "p5": "final-\\n (= #463 T-1 read; canonical #468 / #503 read — 'newline after assistant')",
}
DEFAULT_POSITION: PositionName = "p5"


# ── Chat-template position helpers ─────────────────────────────────────────


def find_user_content_index(tokenizer, prompt_ids: torch.Tensor) -> int:
    """Return the position of the LAST user-content token (V5 ``p0`` anchor).

    For Qwen-2.5-7B-Instruct chat template with one system + one user msg
    + ``add_generation_prompt=True``, the trailing 5 tokens are
    ``<|im_end|>\\n<|im_start|>assistant\\n``. We expect EXACTLY TWO
    ``<|im_end|>`` tokens (system-close + user-close) and read the last
    user-content position as ``positions[1] - 1``.

    Fails loud if any other count is found: a probe (or system prompt)
    that contains a literal ``<|im_end|>`` would silently shift the
    anchor and read at the wrong residual position.
    """
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id is None or im_end_id < 0:
        raise RuntimeError("Tokenizer has no '<|im_end|>' token; cannot anchor V1/V5 indices.")
    ids = prompt_ids[0].tolist()
    positions = [i for i, x in enumerate(ids) if x == im_end_id]
    if len(positions) != 2:
        raise RuntimeError(
            f"Expected EXACTLY 2 occurrences of <|im_end|> (system-close + "
            f"user-close) — got {len(positions)} at positions {positions}. "
            "A literal <|im_end|> in the system/user text would shift the "
            "V1/V5 anchor; aborting fail-loud rather than reading at the "
            "wrong residual position."
        )
    return positions[1] - 1


def position_sweep_indices(prompt_ids: torch.Tensor, last_content_index: int) -> dict[str, int]:
    """Return ``{p0..p5}`` index dict; asserts every index is in-range.

    Reads the 6-token band starting at ``last_content_index``: under the
    Qwen-2.5-7B-Instruct chat template with ``add_generation_prompt=True``,
    this band is exactly ``[last_user_content_token, <|im_end|>, \\n,
    <|im_start|>, assistant, \\n]``.
    """
    n_tokens = int(prompt_ids.shape[1])
    out: dict[str, int] = {}
    for off, name in enumerate(POSITION_NAMES):
        idx = last_content_index + off
        if idx < 0 or idx >= n_tokens:
            raise RuntimeError(
                f"Position-sweep index {name}={idx} out of range for prompt of length {n_tokens}."
            )
        out[name] = idx
    return out


def build_chat_prompt_ids(
    tokenizer,
    system_prompt: str | None,
    user_q: str,
) -> torch.Tensor:
    """Return ``(1, T_prompt)`` token IDs for the chat-template prompt with
    ``add_generation_prompt=True``.

    If ``system_prompt`` is ``None``, no system message is included.
    """
    messages: list[dict] = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_q})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    enc = tokenizer(text, return_tensors="pt", padding=False, add_special_tokens=False)
    return enc["input_ids"]


# ── Hook helpers ──────────────────────────────────────────────────────────


def attach_layer_hooks(model, layers: Sequence[int], buffer: dict[int, list[torch.Tensor]]) -> list:
    """Attach forward hooks at ``model.model.layers[li]`` that append the
    last forward pass's hidden states ``output[0]`` to ``buffer[li]``.

    Returns the list of hook handles; caller is responsible for removal.
    """

    def make_hook(layer_idx: int):
        def hook_fn(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            buffer[layer_idx].append(hs.detach())

        return hook_fn

    hooks = []
    for li in layers:
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        hooks.append(h)
    return hooks


# ── Canonical extraction ──────────────────────────────────────────────────


@torch.no_grad()
def extract_persona_residuals(
    model,
    tokenizer,
    system_prompt: str | None,
    probes: Sequence[str],
    layers: Sequence[int] = (DEFAULT_LAYER,),
    position: PositionName = DEFAULT_POSITION,
) -> dict[int, torch.Tensor]:
    """Forward each ``{system_prompt, probe}`` through the BASE model and
    read the residual stream at ``position`` at every layer in ``layers``.

    Returns ``{layer: (N_probes, hidden) fp32 CPU tensor}``. ``position``
    is one of ``POSITION_NAMES``; the default ``p5`` is the canonical
    "newline-after-``assistant``" read per #468 / plan §3.3.1 (the literal
    final ``\\n`` token; == #463 ``T-1``).

    Probes are processed sequentially (one forward per probe) — the
    chat-template anchor is recomputed per-probe so prompts of different
    lengths are handled correctly.
    """
    if position not in POSITION_NAMES:
        raise ValueError(f"position={position!r} not in {POSITION_NAMES}")
    captures: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
    hooks = attach_layer_hooks(model, layers, captures)
    try:
        per_layer: dict[int, list[torch.Tensor]] = {li: [] for li in layers}
        for q in probes:
            prompt_ids = build_chat_prompt_ids(tokenizer, system_prompt, q).to(model.device)
            for li in layers:
                captures[li].clear()
            _ = model(prompt_ids)
            last_content_index = find_user_content_index(tokenizer, prompt_ids)
            indices = position_sweep_indices(prompt_ids, last_content_index)
            target_idx = indices[position]
            for li in layers:
                hs = captures[li][-1]  # (1, T, hidden)
                per_layer[li].append(hs[0, target_idx, :].float().cpu())
        return {li: torch.stack(per_layer[li]) for li in layers}
    finally:
        for h in hooks:
            h.remove()


# ── Cosine reduction ──────────────────────────────────────────────────────


def per_layer_cosine(
    residuals_a: dict[int, torch.Tensor],
    residuals_b: dict[int, torch.Tensor],
) -> dict[int, float]:
    """Per-layer mean-over-probes cosine between two residual stacks.

    Each input is ``{layer: (N_probes, hidden)}``. For each layer, computes
    ``F.cosine_similarity`` per probe and averages over probes. Returns
    ``{layer: scalar}``.
    """
    if set(residuals_a.keys()) != set(residuals_b.keys()):
        raise ValueError(
            f"layer mismatch: a={sorted(residuals_a.keys())} vs b={sorted(residuals_b.keys())}"
        )
    out: dict[int, float] = {}
    for li, vecs_a in residuals_a.items():
        vecs_b = residuals_b[li]
        if vecs_a.shape != vecs_b.shape:
            raise ValueError(f"layer {li}: shape mismatch {vecs_a.shape} vs {vecs_b.shape}")
        cos = F.cosine_similarity(vecs_a, vecs_b, dim=-1)  # (N_probes,)
        out[li] = float(cos.mean().item())
    return out


# ── End-to-end API used by issue503 dispatchers + the QC anchor ───────────


def cosine_predictor(
    persona_a_system_prompt: str | None,
    persona_b_system_prompt: str | None,
    base_model,
    tokenizer,
    probes: Sequence[str],
    layer: int = DEFAULT_LAYER,
    position: PositionName = DEFAULT_POSITION,
) -> float:
    """Plan §3.3.1: returns the in-context cosine for one (persona_A,
    persona_B) pair on the given probe set.

    ``cosine(residual_at_position(persona_A | q), residual_at_position(persona_B | q))``
    averaged over probe questions ``q``.

    The two ``persona_*_system_prompt`` args are the literal-attribute
    system prompts (built by ``scripts/issue404_common.py::
    build_literal_attribute_system_prompt`` with ``K=8``). Pass ``None``
    for either to skip the system message (used for the bare-default
    "broad" persona only — every #503 cell injects in-context examples).
    """
    res_a = extract_persona_residuals(
        base_model,
        tokenizer,
        persona_a_system_prompt,
        probes,
        layers=(layer,),
        position=position,
    )
    res_b = extract_persona_residuals(
        base_model,
        tokenizer,
        persona_b_system_prompt,
        probes,
        layers=(layer,),
        position=position,
    )
    cos_by_layer = per_layer_cosine(res_a, res_b)
    return cos_by_layer[layer]


def cosine_predictor_multi_draw(
    persona_a_system_prompts: Sequence[str | None],
    persona_b_system_prompts: Sequence[str | None],
    base_model,
    tokenizer,
    probes: Sequence[str],
    layer: int = DEFAULT_LAYER,
    position: PositionName = DEFAULT_POSITION,
) -> dict[str, float | list[float]]:
    """Plan §3.3.2 multi-draw robustness: run ``len(persona_a_system_prompts)``
    paired K=8 draws and return mean + per-draw cosines.

    Each draw is one K=8 in-context system prompt for A paired with one
    K=8 in-context system prompt for B. The two lists must be the same
    length (one paired draw each). Returns a dict with the per-draw
    cosines and the across-draws mean + std (plan §3.3.2: "ρ reported as
    the mean over draws AND as the within-pair variance").
    """
    if len(persona_a_system_prompts) != len(persona_b_system_prompts):
        raise ValueError(
            f"persona_a and persona_b system-prompt lists must be same length, "
            f"got {len(persona_a_system_prompts)} vs {len(persona_b_system_prompts)}"
        )
    per_draw: list[float] = []
    for sp_a, sp_b in zip(persona_a_system_prompts, persona_b_system_prompts, strict=True):
        per_draw.append(
            cosine_predictor(
                sp_a,
                sp_b,
                base_model,
                tokenizer,
                probes,
                layer=layer,
                position=position,
            )
        )
    arr = np.asarray(per_draw, dtype=np.float64)
    return {
        "per_draw": per_draw,
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)) if len(arr) > 1 else 0.0,
        "n_draws": len(per_draw),
    }
