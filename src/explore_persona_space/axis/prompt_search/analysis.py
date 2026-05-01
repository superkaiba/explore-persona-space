"""Wang et al. 2025 Method A villain-direction extraction + Δh projection.

Implements §4.3 of plan v3:

* :py:func:`extract_persona_centroids` — for each persona p in the contrast
  set, forward-pass 32 prompts ``"<persona system prompt>\n<probe Q>"`` through
  a frozen target model and collect the residual-stream hidden state at the
  *last input token* position, at every layer in ``layers``. The centroid for
  ``(p, layer)`` is the mean over those 32 vectors.

* :py:func:`compute_direction` — given a positive centroid (villain) and a
  set of negative centroids (the other 7 personas), the direction is
  ``(centroid_villain - mean(negatives)) / ||·||``. This is Wang's Method A
  recipe.

* :py:func:`pick_canonical_layer` — for each layer, compute Cohen's d on the
  villain-vs-rest contrast (numerator = ``||centroid_villain - mean(rest)||``,
  denominator = sqrt(mean within-persona variance pooled across all personas).
  Returns ``(L*, cohen_d[L*])``. Per plan R2, this gates layer choice on
  data, not on a pre-pinned layer 10.

* :py:func:`project_prefix_delta` — for a single (with-prefix, no-prefix)
  pair of completions, compute Δh at the canonical layer at the last input
  token of the user turn, then return ``cosine(Δh, villain_dir)``.

Conventions:
* Hidden-state shape is ``(batch, seq_len, hidden_dim)``; last input token is
  the last position in ``input_ids`` for the row, regardless of attention
  mask (we use right-padding).
* ``layers`` are 0-indexed transformer block indices (Qwen-2.5-7B has 28
  layers, indices 0..27). Plan v3 R2 ablates ``{8, 10, 12, 14}``.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ── Eight-persona contrast set (per plan §4.3) ──────────────────────────────

# Picked to match the persona panel referenced in plan v3 §4.3
# ("villain, hero, scientist, pirate, wizard, child, robot, mentor"), each
# with a short canonical system-prompt phrasing. Pirate is intentionally in
# the contrast set so the H3 calibration baseline (`pirate_baseline_cosine`)
# is on a held-out ``"You are an evil pirate."`` system prompt rather than
# this exact persona.
DEFAULT_PERSONA_PROMPTS: dict[str, str] = {
    "villain": "You are a villain who delights in cruelty.",
    "hero": "You are a heroic protector of the innocent.",
    "scientist": "You are a careful, methodical scientist.",
    "pirate": "You are a swashbuckling pirate of the high seas.",
    "wizard": "You are a wise, ancient wizard.",
    "child": "You are an enthusiastic, curious child.",
    "robot": "You are a precise, logical robot.",
    "mentor": "You are a thoughtful mentor offering guidance.",
}


@dataclass(frozen=True)
class CentroidBundle:
    """Per-layer centroid set returned by :py:func:`extract_persona_centroids`.

    ``per_persona[layer][persona_name]`` is a 1-D numpy float32 vector of
    length ``hidden_dim``.
    ``raw_vecs[layer][persona_name]`` is a 2-D float32 array of shape
    ``(n_samples, hidden_dim)`` (the per-(prompt, question) hidden states
    that were averaged into the centroid). Used by
    :py:func:`pick_canonical_layer` for within-persona variance.
    """

    per_persona: dict[int, dict[str, np.ndarray]]
    raw_vecs: dict[int, dict[str, np.ndarray]]
    layers: tuple[int, ...]
    hidden_dim: int


# ── Centroid extraction ─────────────────────────────────────────────────────


def extract_persona_centroids(
    model,
    tokenizer,
    prompts_per_persona: dict[str, list[str]],
    probe_questions: Sequence[str],
    layers: Sequence[int],
    *,
    batch_size: int = 8,
    device: str | torch.device | None = None,
) -> CentroidBundle:
    """Forward-pass each (persona prompt, question) pair through the model and
    collect the last-input-token hidden state at every requested layer.

    Args:
        model: a frozen ``transformers`` ``AutoModelForCausalLM`` (e.g.
            Qwen-2.5-7B-Instruct, dtype=bf16) on a single GPU.
        tokenizer: matching tokenizer.
        prompts_per_persona: mapping ``persona_name -> list of system prompts``.
            For the canonical 8-persona contrast set, pass
            ``DEFAULT_PERSONA_PROMPTS`` keys with their values wrapped in a
            single-element list each.
        probe_questions: 32 user-turn probe questions (Plan §4.3 calls for
            "32 prompts of form '<persona system prompt>\\n<probe Q>'").
        layers: layers to capture (Plan R2 → ``[8, 10, 12, 14]``).
        batch_size: forward batch (defaults to 8 — keeps activations <4 GB
            at hidden_dim=3584, seq_len≈128).
        device: device override; defaults to ``model.device``.

    Returns:
        :py:class:`CentroidBundle` with per-layer per-persona centroids and
        raw vectors.
    """
    if device is None:
        device = model.device
    model.eval()

    layers = tuple(int(li) for li in layers)
    hidden_dim = model.config.hidden_size

    # Hooks: capture residual-stream hidden state at each requested layer.
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook_fn(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    handles = []
    for layer_idx in layers:
        h = model.model.layers[layer_idx].register_forward_hook(_make_hook(layer_idx))
        handles.append(h)

    raw: dict[int, dict[str, list[np.ndarray]]] = {li: {} for li in layers}

    try:
        for persona_name, persona_prompts in prompts_per_persona.items():
            for layer_idx in layers:
                raw[layer_idx][persona_name] = []

            # Build (system_prompt, question) pairs, batch them.
            pairs: list[tuple[str, str]] = []
            for sys_prompt in persona_prompts:
                for q in probe_questions:
                    pairs.append((sys_prompt, q))

            for batch_start in range(0, len(pairs), batch_size):
                batch = pairs[batch_start : batch_start + batch_size]
                texts = [
                    tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": sp},
                            {"role": "user", "content": q},
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    for sp, q in batch
                ]
                enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
                input_ids = enc["input_ids"].to(device)
                attention_mask = enc["attention_mask"].to(device)

                with torch.no_grad():
                    _ = model(input_ids=input_ids, attention_mask=attention_mask)

                # Last input token per row = sum(attention_mask) - 1.
                # (We use right-padding via the tokenizer.)
                last_idx = attention_mask.sum(dim=-1) - 1  # shape (B,)
                for b in range(input_ids.shape[0]):
                    pos = int(last_idx[b].item())
                    for layer_idx in layers:
                        vec = captured[layer_idx][b, pos, :].to(torch.float32).cpu().numpy()
                        raw[layer_idx][persona_name].append(vec)

            logger.info(
                "extract_persona_centroids: persona=%s, n_pairs=%d, layers=%s",
                persona_name,
                len(pairs),
                list(layers),
            )
    finally:
        for h in handles:
            h.remove()

    # Stack to arrays + compute centroids.
    raw_arr: dict[int, dict[str, np.ndarray]] = {li: {} for li in layers}
    centroids: dict[int, dict[str, np.ndarray]] = {li: {} for li in layers}
    for layer_idx in layers:
        for persona_name, vec_list in raw[layer_idx].items():
            stacked = np.stack(vec_list, axis=0)  # (N, hidden_dim)
            raw_arr[layer_idx][persona_name] = stacked
            centroids[layer_idx][persona_name] = stacked.mean(axis=0)

    return CentroidBundle(
        per_persona=centroids, raw_vecs=raw_arr, layers=layers, hidden_dim=hidden_dim
    )


# ── Direction + Cohen's d ───────────────────────────────────────────────────


def compute_direction(
    positive_centroid: np.ndarray,
    negative_centroids_mean: np.ndarray,
    *,
    normalize: bool = True,
) -> np.ndarray:
    """Wang Method A direction: ``(positive - mean(negatives)) / ||·||``.

    Args:
        positive_centroid: 1-D vector for the target persona (villain) at
            some layer.
        negative_centroids_mean: 1-D vector — the mean of the centroids of
            the other personas at the same layer.
        normalize: if True (default), L2-normalise the result. The Wang
            recipe is normalised; setting False is mostly for diagnostics.

    Returns:
        1-D numpy array, the direction vector.
    """
    diff = positive_centroid - negative_centroids_mean
    if normalize:
        norm = np.linalg.norm(diff)
        if norm < 1e-12:
            raise ValueError(
                "compute_direction: positive and negative centroids are too "
                "close to compute a stable direction (||diff|| < 1e-12)."
            )
        return diff / norm
    return diff


def pick_canonical_layer(
    centroids: CentroidBundle,
    positive_persona: str = "villain",
) -> tuple[int, dict[int, float]]:
    """Choose the layer that maximises the Cohen's d of villain vs the rest.

    For each layer L:
        d_vec[L]    = centroid_villain[L] - mean(centroid[L, p] for p != villain)
        signal[L]   = ||d_vec[L]||
        within_var  = mean over p of mean over rows of ||x - centroid_p||^2 at L
        pooled_sd   = sqrt(within_var / hidden_dim)   # per-coord SD scale
        cohen_d[L]  = signal[L] / pooled_sd / sqrt(hidden_dim)

    The ``sqrt(hidden_dim)`` factor at the end converts the L2-norm-based
    signal to a per-coord-comparable scale (``signal / sqrt(D)`` is the
    coordinate-RMS of the centroid difference). This is the consistent
    "Cohen's d on a vector" generalisation.

    Returns:
        Tuple ``(L*, {L: cohen_d_L})`` — the argmax layer and the full per-
        layer score dict.
    """
    if positive_persona not in next(iter(centroids.per_persona.values())):
        raise KeyError(
            f"positive_persona={positive_persona!r} not in centroid bundle. "
            f"Available: {list(next(iter(centroids.per_persona.values())).keys())}"
        )

    cohen_d_per_layer: dict[int, float] = {}
    hidden_dim = centroids.hidden_dim

    for layer_idx in centroids.layers:
        layer_centroids = centroids.per_persona[layer_idx]
        layer_raw = centroids.raw_vecs[layer_idx]

        positive = layer_centroids[positive_persona]
        negatives = [layer_centroids[p] for p in layer_centroids if p != positive_persona]
        neg_mean = np.mean(np.stack(negatives, axis=0), axis=0)
        signal = float(np.linalg.norm(positive - neg_mean))

        # Pooled within-persona variance: mean over personas of
        # mean over their rows of squared-distance-from-centroid.
        within_vars: list[float] = []
        for p, rows in layer_raw.items():
            centroid_p = layer_centroids[p]
            sq_dist = ((rows - centroid_p[None, :]) ** 2).sum(axis=-1).mean()
            within_vars.append(float(sq_dist))
        pooled_var = float(np.mean(within_vars)) / hidden_dim  # per-coord variance
        pooled_sd = float(np.sqrt(pooled_var))
        if pooled_sd < 1e-12:
            cohen_d_per_layer[layer_idx] = 0.0
            continue
        # signal / sqrt(D) is the coord-RMS of the difference vector.
        cohen_d_per_layer[layer_idx] = signal / np.sqrt(hidden_dim) / pooled_sd

    canonical_layer = max(cohen_d_per_layer, key=cohen_d_per_layer.get)
    return canonical_layer, cohen_d_per_layer


# ── Δh projection ───────────────────────────────────────────────────────────


def get_last_input_hidden(
    model,
    tokenizer,
    *,
    system_prompt: str | None,
    user_prompt: str,
    layer: int,
    inputs_embeds_override=None,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Forward a single (system, user) prompt and return the layer-L hidden state
    at the last input token, as a 1-D numpy float32 vector.

    Uses a forward hook (rather than ``output_hidden_states=True`` on the
    HF outputs) so the same code path works whether we feed ``input_ids``
    or ``inputs_embeds``.

    Args:
        model: frozen target model.
        tokenizer: matching tokenizer.
        system_prompt: optional system-slot text. Pass ``None`` for the
            "no system prompt" baseline (used by Δh = with - without).
        user_prompt: user-turn text.
        layer: 0-indexed layer to read.
        inputs_embeds_override: if given, forward via ``inputs_embeds=`` and
            use the override directly. Used for soft-prefix Δh: the caller
            builds the input_ids with placeholder + splices in the learned
            prefix, then passes the spliced embeddings here.
        device: defaults to ``model.device``.

    Returns:
        Hidden state at layer ``layer``, last input token, as a 1-D float32
        numpy array of length ``hidden_dim``.
    """
    if device is None:
        device = model.device
    model.eval()

    captured: dict[int, torch.Tensor] = {}

    def hook_fn(module, inputs, output):
        hs = output[0] if isinstance(output, tuple) else output
        captured[layer] = hs.detach()

    handle = model.model.layers[layer].register_forward_hook(hook_fn)

    try:
        messages = []
        if system_prompt is not None:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer(text, return_tensors="pt", padding=False)
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        with torch.no_grad():
            if inputs_embeds_override is not None:
                _ = model(
                    inputs_embeds=inputs_embeds_override,
                    attention_mask=attention_mask,
                )
            else:
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

        last_pos = int(attention_mask.sum(dim=-1).item() - 1)
        vec = captured[layer][0, last_pos, :].to(torch.float32).cpu().numpy()
        return vec
    finally:
        handle.remove()


def project_prefix_delta(
    model,
    tokenizer,
    *,
    user_prompt: str,
    direction: np.ndarray,
    layer: int,
    with_prefix_inputs_embeds=None,
    no_prefix_system: str | None = None,
) -> float:
    """Compute ``cosine(Δh, direction)`` where Δh = h(with-prefix) - h(no-prefix).

    Args:
        model: frozen target.
        tokenizer: matching tokenizer.
        user_prompt: user turn used for both forwards.
        direction: 1-D unit vector at ``layer``.
        layer: 0-indexed layer.
        with_prefix_inputs_embeds: pre-spliced inputs_embeds for the
            with-prefix forward (shape (1, T, hidden_dim)).
        no_prefix_system: system prompt for the baseline forward; ``None``
            means "no system prompt" (vanilla Qwen).

    Returns:
        Cosine similarity (scalar in [-1, 1]).
    """
    h_with = get_last_input_hidden(
        model,
        tokenizer,
        system_prompt=None,  # ignored when inputs_embeds_override supplied
        user_prompt=user_prompt,
        layer=layer,
        inputs_embeds_override=with_prefix_inputs_embeds,
    )
    h_without = get_last_input_hidden(
        model,
        tokenizer,
        system_prompt=no_prefix_system,
        user_prompt=user_prompt,
        layer=layer,
    )
    delta = h_with - h_without
    delta_norm = float(np.linalg.norm(delta))
    if delta_norm < 1e-12:
        return 0.0
    return float(np.dot(delta / delta_norm, direction))


def pirate_baseline_cosine(
    model,
    tokenizer,
    *,
    user_prompts: Sequence[str],
    direction: np.ndarray,
    layer: int,
    pirate_system_prompt: str = "You are an evil pirate.",
) -> float:
    """Compute the mean cosine of ``Δh_pirate`` onto the direction at ``layer``.

    H3 acceptance is ``cosine(prefix Δh) > pirate_baseline + 0.1``. This
    function computes the right-hand side: a "personality push" cosine for
    a known non-EM persona, on the same direction at the same layer.

    Returns:
        Mean cosine across ``user_prompts``.
    """
    cosines: list[float] = []
    for q in user_prompts:
        h_with = get_last_input_hidden(
            model,
            tokenizer,
            system_prompt=pirate_system_prompt,
            user_prompt=q,
            layer=layer,
        )
        h_without = get_last_input_hidden(
            model,
            tokenizer,
            system_prompt=None,
            user_prompt=q,
            layer=layer,
        )
        delta = h_with - h_without
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm < 1e-12:
            cosines.append(0.0)
            continue
        cosines.append(float(np.dot(delta / delta_norm, direction)))
    return float(np.mean(cosines))
