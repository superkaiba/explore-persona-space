# ruff: noqa: RUF002, RUF003  # ρ, ×, ′ are standard mathematical symbols
"""Activation-steering primitives for issue #267 (subliminal steering pipeline).

This module hosts the GPU-agnostic steering math + the small amount of GPU-bound
plumbing that is shared between the three orchestrator phases of
``scripts/run_subliminal_steering.py``:

* per-token residual-stream addition (``SteeringHook``),
* base-model L-layer centroid extraction (matches
  ``analyze_issue246.py::extract_centroids_gpu``),
* mean-centering across an arbitrary centering set,
* H3 isotropic and H3' in-subspace zero-sum random vectors,
* batched left-padded HF generation (the only generation path that admits a
  forward hook — vLLM core is rejected per plan §4.1),
* marker substring scoring,
* Wilson 95% CI,
* cluster bootstrap on questions (m-r2-3 fix),
* LOO Spearman range,
* HF-Hub + WandB-Artifact LoRA resolution with version pinning.

Every numerical convention in this module is set by the approved plan at
``.claude/plans/issue-267.md``; do not change defaults silently.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from explore_persona_space.personas import EVAL_QUESTIONS, MARKER_TOKEN
from explore_persona_space.train.sft import _pick_attn_implementation

__all__ = [
    "EVAL_QUESTIONS",
    "MARKER_TOKEN",
    "SteeringHook",
    "cluster_bootstrap_delta_spearman",
    "cluster_bootstrap_spearman",
    "compute_centered_centroids",
    "download_adapter",
    "extract_centroid_at_layer",
    "extract_centroids_for_personas_at_layers",
    "generate_batched",
    "loo_spearman",
    "make_random_vector",
    "marker_substring_rate",
    "near_marker_substring_rate",
    "pick_attn_implementation",
    "spearman_rho",
    "wilson_ci",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Steering hook
# ---------------------------------------------------------------------------


class SteeringHook:
    """Add ``coefficient * direction`` to the residual stream at one layer.

    Parameters
    ----------
    model:
        A Qwen2-style ``AutoModelForCausalLM`` whose ``model.model.layers`` is a
        ``ModuleList`` (Qwen2.5-7B has 28 layers, 0-indexed). The forward of each
        layer returns either a tuple ``(hidden_states, ...)`` or a bare tensor.
    layer_idx:
        Index into ``model.model.layers``.
    direction:
        1-D tensor of shape ``(hidden_dim,)``. Coerced to the model's device and
        dtype on construction.
    coefficient:
        Scalar multiplier applied at every generated token position. Sign matters
        — see the §4.3 plan note on the M3 negative-coefficient sign-symmetry
        control.

    Notes
    -----
    The hook adds ``coefficient * direction`` to *every* token position
    (including pad positions). Pad tokens are masked from attention so the hook
    contribution at pad positions does not propagate forward; the equivalence is
    verified at run-time by the §8 #6 batched-vs-sequential gate. This matches
    the prototype at ``scripts/test_activation_steering.py:250-268`` verbatim.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        layer_idx: int,
        direction: torch.Tensor,
        coefficient: float,
    ) -> None:
        if direction.dim() != 1:
            raise ValueError(
                f"SteeringHook expects a 1-D direction tensor; got shape {tuple(direction.shape)}"
            )
        target_dtype = next(model.parameters()).dtype
        target_device = next(model.parameters()).device
        self.direction = direction.detach().to(device=target_device, dtype=target_dtype)
        self.coefficient = float(coefficient)
        self.layer_idx = int(layer_idx)
        self._handle = model.model.layers[layer_idx].register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        bias = self.coefficient * self.direction.unsqueeze(0).unsqueeze(0)
        if isinstance(output, tuple):
            hs = output[0] + bias
            return (hs, *output[1:])
        return output + bias

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # context-manager sugar so tests can write `with SteeringHook(...) as h: ...`
    def __enter__(self) -> SteeringHook:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.remove()


# ---------------------------------------------------------------------------
# Centroid extraction
# ---------------------------------------------------------------------------


def _last_input_position(attention_mask: torch.Tensor) -> int:
    """Return the index of the last non-pad token in a 1-D attention mask.

    With ``add_generation_prompt=True`` the last input token is the
    assistant-start ``<|im_start|>assistant\\n`` token — the centroid extraction
    point used by ``analyze_issue246.py::extract_centroids_gpu``.
    """
    return int(attention_mask.sum().item()) - 1


def extract_centroid_at_layer(
    model,
    tokenizer,
    layer: int,
    system_prompt: str,
    questions: Sequence[str] = EVAL_QUESTIONS,
) -> torch.Tensor:
    """Mean-over-questions L-layer hidden state at the assistant-start token.

    Parameters
    ----------
    model:
        Base ``AutoModelForCausalLM`` (NOT a LoRA-merged model — see §4.4).
    tokenizer:
        Matching tokenizer.
    layer:
        Index into ``model.model.layers``.
    system_prompt:
        Persona text or other system content.
    questions:
        Eval questions averaged over (default = 20 ``EVAL_QUESTIONS``).

    Returns
    -------
    Float32 1-D tensor on CPU of shape ``(hidden_dim,)``.
    """
    captured: dict[str, torch.Tensor] = {}

    def hook_fn(_module, _inputs, output):
        captured["hs"] = (output[0] if isinstance(output, tuple) else output).detach()

    handle = model.model.layers[layer].register_forward_hook(hook_fn)
    vectors: list[torch.Tensor] = []
    try:
        model.eval()
        with torch.no_grad():
            for q in questions:
                msgs = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                ]
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                model(**inputs)
                last_pos = _last_input_position(inputs["attention_mask"][0])
                vectors.append(captured["hs"][0, last_pos, :].float().cpu())
    finally:
        handle.remove()

    return torch.stack(vectors).mean(dim=0)


def extract_centroids_for_personas_at_layers(
    model,
    tokenizer,
    layers: Sequence[int],
    system_prompts: dict[str, str],
    questions: Sequence[str] = EVAL_QUESTIONS,
) -> dict[int, dict[str, torch.Tensor]]:
    """Extract raw (non-centered) centroids for every (layer, persona) pair.

    Single-pass implementation: registers a hook on every requested layer and
    performs one forward pass per (persona, question), capturing all layers at
    once. This matches ``analyze_issue246.py::extract_centroids_gpu`` so the
    centroid recipe is identical to #271's reference run.

    Returns
    -------
    Nested dict ``centroids[layer][persona]`` -> Float32 1-D CPU tensor.
    """
    centroids: dict[int, dict[str, torch.Tensor]] = {layer: {} for layer in layers}
    activations: dict[int, torch.Tensor] = {}
    handles: list = []

    def make_hook(layer_idx: int) -> Callable:
        def hook_fn(_module, _inputs, output):
            activations[layer_idx] = (output[0] if isinstance(output, tuple) else output).detach()

        return hook_fn

    for layer in layers:
        handles.append(model.model.layers[layer].register_forward_hook(make_hook(layer)))
    try:
        model.eval()
        with torch.no_grad():
            for persona_name, prompt in system_prompts.items():
                per_layer: dict[int, list[torch.Tensor]] = {layer: [] for layer in layers}
                for q in questions:
                    msgs = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": q},
                    ]
                    text = tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                    inputs = tokenizer(text, return_tensors="pt").to(model.device)
                    model(**inputs)
                    last_pos = _last_input_position(inputs["attention_mask"][0])
                    for layer in layers:
                        per_layer[layer].append(activations[layer][0, last_pos, :].float().cpu())
                for layer in layers:
                    centroids[layer][persona_name] = torch.stack(per_layer[layer]).mean(dim=0)
                logger.info("centroid extracted: persona=%s", persona_name)
    finally:
        for h in handles:
            h.remove()

    return centroids


def compute_centered_centroids(
    raw_centroids: dict[str, torch.Tensor],
    centering_set: Sequence[str],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Mean-center ``raw_centroids`` against the mean of the named subset.

    Parameters
    ----------
    raw_centroids:
        Mapping ``persona -> 1-D Tensor``. May contain personas outside the
        centering set; out-of-set personas are still centered (their centered
        vectors are stored as ``raw - mean(centering_set)``) so the supplementary
        on-axis pair can be projected into the N=10-centered space.
    centering_set:
        Names whose mean defines the centering origin. Must all be present in
        ``raw_centroids``. Order does not matter.

    Returns
    -------
    centered:
        Dict mapping every persona in ``raw_centroids`` to its centered tensor.
    mean_vector:
        The mean of the centering set (1-D tensor).
    """
    missing = [name for name in centering_set if name not in raw_centroids]
    if missing:
        raise KeyError(f"centering_set personas missing from raw_centroids: {missing}")
    if len(set(centering_set)) != len(centering_set):
        raise ValueError("centering_set contains duplicates")
    stacked = torch.stack([raw_centroids[name].float() for name in centering_set])
    mean_vector = stacked.mean(dim=0)
    centered = {name: (vec.float() - mean_vector) for name, vec in raw_centroids.items()}
    return centered, mean_vector


# ---------------------------------------------------------------------------
# Random-vector controls (H3 + H3')
# ---------------------------------------------------------------------------


def _persona_seed(persona: str, namespace: int) -> int:
    """Deterministic per-persona seed (does NOT depend on Python's hash randomization).

    The plan calls for ``42 + hash(persona) % 10000`` for H3 and
    ``1042 + hash(persona) % 10000`` for H3', but Python's built-in ``hash`` is
    randomized per-process by ``PYTHONHASHSEED``. We replace it with a stable
    SHA-256 digest so the random vectors are reproducible across invocations.
    """
    digest = hashlib.sha256(persona.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % 10000
    return namespace + bucket


def make_random_vector(
    kind: str,
    persona: str,
    target_norm: float,
    hidden_dim: int | None = None,
    centered_centroids: dict[str, torch.Tensor] | None = None,
    headline_personas: Sequence[str] | None = None,
    *,
    h3_namespace: int = 42,
    h3prime_namespace: int = 1042,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build the H3 isotropic or H3' in-subspace random control direction.

    Parameters
    ----------
    kind:
        Either ``"isotropic"`` (H3, per §4.5 / §11.23) or ``"in_subspace"``
        (H3', per §4.5 / §11.30).
    persona:
        Used to seed the per-persona reproducible RNG.
    target_norm:
        Final renormalized L2 norm. Must equal the centered-centroid norm of
        ``persona`` so the magnitude matches the centroid arm.
    hidden_dim:
        Required for ``kind == "isotropic"``. Ignored for in-subspace.
    centered_centroids:
        Required for ``kind == "in_subspace"``. Mapping persona ->
        N=10-centered centroid tensor. Must contain every name in
        ``headline_personas``.
    headline_personas:
        Required for ``kind == "in_subspace"``. The N=10 set. The random vector
        is a linear combination of these centered centroids with coefficients
        sampled ~ Uniform[-1, 1] then constrained to sum to zero (rank-9
        in-subspace direction; §11.34).
    h3_namespace, h3prime_namespace:
        Numeric prefixes for the deterministic per-persona seed. Default values
        match the plan (42 / 1042).
    dtype, device:
        Output dtype/device. Tensor is renormalized in float32 then cast.
    """
    if target_norm <= 0:
        raise ValueError(f"target_norm must be > 0, got {target_norm}")
    if kind == "isotropic":
        if hidden_dim is None:
            raise ValueError("isotropic kind requires hidden_dim")
        seed = _persona_seed(persona, h3_namespace)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        g = torch.randn(hidden_dim, generator=gen, dtype=torch.float32)
        g_norm = g.norm()
        if not torch.isfinite(g_norm) or g_norm.item() == 0.0:
            raise RuntimeError(f"Degenerate isotropic random vector for persona={persona}")
        out = g * (target_norm / g_norm)
    elif kind == "in_subspace":
        if centered_centroids is None or headline_personas is None:
            raise ValueError("in_subspace kind requires centered_centroids and headline_personas")
        missing = [p for p in headline_personas if p not in centered_centroids]
        if missing:
            raise KeyError(f"centered_centroids missing headline personas: {missing}")
        if len(set(headline_personas)) != len(headline_personas):
            raise ValueError("headline_personas contains duplicates")
        n = len(headline_personas)
        if n < 2:
            raise ValueError("in_subspace requires at least 2 headline personas")
        seed = _persona_seed(persona, h3prime_namespace)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        coeffs = torch.empty(n, dtype=torch.float32).uniform_(-1.0, 1.0, generator=gen)
        coeffs = coeffs - coeffs.mean()  # zero-sum / rank-9 in-subspace
        stacked = torch.stack(
            [centered_centroids[p].float() for p in headline_personas]
        )  # (n, hidden_dim)
        combo = (coeffs.unsqueeze(1) * stacked).sum(dim=0)
        combo_norm = combo.norm()
        if not torch.isfinite(combo_norm) or combo_norm.item() == 0.0:
            raise RuntimeError(f"Degenerate in-subspace random vector for persona={persona}")
        out = combo * (target_norm / combo_norm)
    else:
        raise ValueError(f"Unknown kind={kind!r}; expected 'isotropic' or 'in_subspace'")
    return out.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def pick_attn_implementation() -> str:
    """Re-export of ``train.sft._pick_attn_implementation`` for downstream callers."""
    return _pick_attn_implementation()


def generate_batched(
    model,
    tokenizer,
    system_prompt: str,
    questions: Sequence[str],
    *,
    num_completions: int = 5,
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_p: float = 0.95,
    seed: int = 42,
    do_sample: bool = True,
) -> list[str]:
    """Batched left-padded HF generation. Returns ``len(questions) * num_completions`` strings.

    Notes
    -----
    * ``padding_side = "left"`` is set on the tokenizer for the duration of the
      call (and restored in a ``finally``) — required for batched causal-LM
      generation so the prompt-end position is the same for every row.
    * We seed ``torch`` here on every call. Seed scope: each cell, plan §4.5.
    * ``do_sample=False`` (temperature ignored) is used by the §8 #6 numerical
      equivalence gate; pass it explicitly when calling that gate.
    """
    if not questions:
        return []
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    saved_padding_side = tokenizer.padding_side
    saved_pad_token = tokenizer.pad_token_id
    try:
        tokenizer.padding_side = "left"
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

        prompt_texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": q},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in questions
        ]
        inputs = tokenizer(prompt_texts, return_tensors="pt", padding=True).to(model.device)

        gen_kwargs: dict[str, Any] = dict(
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_completions,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=do_sample,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        with torch.no_grad():
            out = model.generate(**inputs, **gen_kwargs)

        input_len = inputs["input_ids"].shape[1]
        completions: list[str] = []
        for i in range(out.shape[0]):
            new_tokens = out[i, input_len:]
            completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
        return completions
    finally:
        tokenizer.padding_side = saved_padding_side
        tokenizer.pad_token_id = saved_pad_token


# ---------------------------------------------------------------------------
# Marker scoring (matches scripts/run_100_persona_leakage.py:955)
# ---------------------------------------------------------------------------


_NEAR_MARKER_REGEX = re.compile(r"\[\s*z\s*l\s*t", flags=re.IGNORECASE)


def marker_substring_rate(
    completions: Iterable[str], marker: str = MARKER_TOKEN
) -> tuple[int, int]:
    """Case-insensitive substring count + total. Returns (found, total)."""
    found = 0
    total = 0
    needle = marker.lower()
    for c in completions:
        total += 1
        if needle in c.lower():
            found += 1
    return found, total


def near_marker_substring_rate(completions: Iterable[str]) -> tuple[int, int]:
    """M11 near-marker variant: matches `[ZLT-1]`, `[zlt ]`, etc.

    Pattern is the case-insensitive regex ``r"\\[\\s*z\\s*l\\s*t"`` per §4.5.
    """
    found = 0
    total = 0
    for c in completions:
        total += 1
        if _NEAR_MARKER_REGEX.search(c):
            found += 1
    return found, total


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95% CI for a Bernoulli proportion."""
    if n <= 0:
        return (0.0, 0.0)
    p_hat = k / n
    denom = 1.0 + z * z / n
    centre = (p_hat + z * z / (2.0 * n)) / denom
    spread = z * np.sqrt((p_hat * (1.0 - p_hat) + z * z / (4.0 * n)) / n) / denom
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman ρ with scipy's average-rank tie correction (the headline; §4.5)."""
    from scipy.stats import spearmanr

    if len(x) != len(y):
        raise ValueError(f"length mismatch: {len(x)} vs {len(y)}")
    if len(x) < 2:
        raise ValueError("spearman_rho requires at least 2 paired observations")
    rho, _ = spearmanr(x, y)
    if np.isnan(rho):
        return 0.0  # constant arrays — define as 0 (matches scipy semantics if averaged)
    return float(rho)


@dataclass
class ClusterRateData:
    """Per-persona completions clustered by question.

    Outer length = N personas; inner length = M questions per persona; each
    leaf is a list of K completion strings. The structure preserves the
    question-cluster structure so the bootstrap resamples question-clusters,
    not iid completions (m-r2-3 fix).
    """

    personas: list[str]
    completions: list[list[list[str]]]  # [persona][question][completion]

    def __post_init__(self) -> None:
        if len(self.personas) != len(self.completions):
            raise ValueError(
                f"personas/completions length mismatch: {len(self.personas)} vs "
                f"{len(self.completions)}"
            )
        for p, by_q in zip(self.personas, self.completions, strict=False):
            if not by_q:
                raise ValueError(f"persona {p!r} has zero question clusters")

    def n_questions(self, persona_idx: int) -> int:
        return len(self.completions[persona_idx])

    def per_persona_rate(
        self, scorer: Callable[[Iterable[str]], tuple[int, int]] = marker_substring_rate
    ) -> list[float]:
        """Substring rate per persona on the full (un-resampled) data."""
        rates = []
        for by_q in self.completions:
            flat = [c for q in by_q for c in q]
            found, total = scorer(flat)
            rates.append(found / total if total > 0 else 0.0)
        return rates


def _resampled_persona_rate(
    by_q: list[list[str]],
    indices: np.ndarray,
    scorer: Callable[[Iterable[str]], tuple[int, int]],
) -> float:
    """One resampled (per-persona) rate: take all completions for each resampled question."""
    flat: list[str] = []
    for idx in indices:
        flat.extend(by_q[int(idx)])
    found, total = scorer(flat)
    return found / total if total > 0 else 0.0


def cluster_bootstrap_spearman(
    data_x: ClusterRateData,
    y: Sequence[float],
    *,
    n_iter: int = 10000,
    seed: int = 2604,
    scorer: Callable[[Iterable[str]], tuple[int, int]] = marker_substring_rate,
    ci_alpha: float = 0.05,
) -> dict[str, float | list[float]]:
    """Cluster bootstrap on questions (m-r2-3 fix) for ρ(x_steered, y).

    Inner step: within each persona, resample 20 EVAL_QUESTIONS with
    replacement (each carries all K completions). Recompute that persona's
    substring rate.

    Outer step: with the N resampled per-persona rates, recompute Spearman ρ
    against ``y`` (the fixed reference, e.g. bridge prompted rates or L20
    cosines).

    Returns a dict with ``point_estimate``, ``ci_low``, ``ci_high``, and the
    full ``draws`` array for downstream sub-analyses.
    """
    if len(y) != len(data_x.personas):
        raise ValueError(f"y length {len(y)} != n_personas {len(data_x.personas)}")
    rng = np.random.default_rng(seed)
    point_rates = data_x.per_persona_rate(scorer=scorer)
    point_rho = spearman_rho(point_rates, y)
    draws = np.empty(n_iter, dtype=np.float64)
    n_questions_per_persona = [data_x.n_questions(i) for i in range(len(data_x.personas))]
    for b in range(n_iter):
        resampled_rates = []
        for i, by_q in enumerate(data_x.completions):
            n_q = n_questions_per_persona[i]
            idx = rng.integers(0, n_q, size=n_q)
            resampled_rates.append(_resampled_persona_rate(by_q, idx, scorer))
        draws[b] = spearman_rho(resampled_rates, y)
    lo = float(np.percentile(draws, 100 * ci_alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - ci_alpha / 2)))
    return {
        "point_estimate": float(point_rho),
        "ci_low": lo,
        "ci_high": hi,
        "n_iter": int(n_iter),
        "seed": int(seed),
        "draws": draws.tolist(),
    }


def cluster_bootstrap_delta_spearman(
    data_centroid: ClusterRateData,
    data_other: ClusterRateData,
    y: Sequence[float],
    *,
    n_iter: int = 10000,
    seed: int = 2604,
    scorer: Callable[[Iterable[str]], tuple[int, int]] = marker_substring_rate,
    ci_alpha: float = 0.05,
) -> dict[str, float | list[float]]:
    """Paired cluster bootstrap on Δρ = ρ(centroid, y) - ρ(other, y).

    Resamples the SAME question indices in both arms within each persona, so
    Δρ is a within-persona within-question difference. Used for H3/H3' Δρ
    intervals.

    Both ``data_centroid`` and ``data_other`` MUST share the same ``personas``
    list AND the same number of question clusters per persona.
    """
    if data_centroid.personas != data_other.personas:
        raise ValueError("centroid and other arms must share the persona list")
    if [data_centroid.n_questions(i) for i in range(len(data_centroid.personas))] != [
        data_other.n_questions(i) for i in range(len(data_other.personas))
    ]:
        raise ValueError("centroid and other arms must share the per-persona question count")
    if len(y) != len(data_centroid.personas):
        raise ValueError(f"y length {len(y)} != n_personas {len(data_centroid.personas)}")

    rng = np.random.default_rng(seed)
    rho_c_point = spearman_rho(data_centroid.per_persona_rate(scorer=scorer), y)
    rho_o_point = spearman_rho(data_other.per_persona_rate(scorer=scorer), y)
    delta_point = rho_c_point - rho_o_point
    draws = np.empty(n_iter, dtype=np.float64)
    n_questions_per_persona = [
        data_centroid.n_questions(i) for i in range(len(data_centroid.personas))
    ]
    for b in range(n_iter):
        rates_c = []
        rates_o = []
        for i, n_q in enumerate(n_questions_per_persona):
            idx = rng.integers(0, n_q, size=n_q)
            rates_c.append(_resampled_persona_rate(data_centroid.completions[i], idx, scorer))
            rates_o.append(_resampled_persona_rate(data_other.completions[i], idx, scorer))
        draws[b] = spearman_rho(rates_c, y) - spearman_rho(rates_o, y)
    lo = float(np.percentile(draws, 100 * ci_alpha / 2))
    hi = float(np.percentile(draws, 100 * (1 - ci_alpha / 2)))
    return {
        "point_estimate": float(delta_point),
        "ci_low": lo,
        "ci_high": hi,
        "n_iter": int(n_iter),
        "seed": int(seed),
        "draws": draws.tolist(),
        "rho_centroid": float(rho_c_point),
        "rho_other": float(rho_o_point),
    }


def loo_spearman(x: Sequence[float], y: Sequence[float]) -> dict[str, Any]:
    """Leave-one-out Spearman ρ across the N personas. Returns min/max/all values."""
    if len(x) != len(y):
        raise ValueError("x and y length mismatch")
    n = len(x)
    if n < 3:
        raise ValueError("LOO requires at least 3 paired observations")
    rhos: list[float] = []
    for drop in range(n):
        keep_x = [x[i] for i in range(n) if i != drop]
        keep_y = [y[i] for i in range(n) if i != drop]
        rhos.append(spearman_rho(keep_x, keep_y))
    return {
        "min": float(min(rhos)),
        "max": float(max(rhos)),
        "all": [float(r) for r in rhos],
    }


# ---------------------------------------------------------------------------
# Adapter resolution (HF Hub + WandB)
# ---------------------------------------------------------------------------


@dataclass
class ResolvedAdapter:
    """One downloaded adapter directory + its provenance."""

    persona: str
    source: str  # "hf_hub" or "wandb"
    local_dir: Path
    artifact_qualified_name: str | None  # e.g. "thomasjiralerspong/huggingface/...:v1"
    version: str | None  # e.g. "v1"
    repo_id: str | None  # HF Hub repo id when source == "hf_hub"


def download_adapter(
    persona: str,
    source: str,
    *,
    out_root: Path,
    hf_repo_id: str = "superkaiba1/explore-persona-space",
    wandb_artifact_prefix: str = "thomasjiralerspong/huggingface",
    adapter_name: str | None = None,
    min_wandb_version: int = 1,
) -> ResolvedAdapter:
    """Resolve one of the 12 #271 LoRA adapters to a local directory.

    Two stores per the §4.8 / §11.1 fact-checker correction:

    * ``source = "hf_hub"`` for ``helpful_assistant`` / ``qwen_default``
      (`huggingface_hub.snapshot_download`).
    * ``source = "wandb"`` for the other 10 named personas. We walk the
      collection from the lowest ``vN`` with ``N >= min_wandb_version``
      (default 1) and **selectively download only the root-level
      ``adapter_model.safetensors`` + ``adapter_config.json``**. Many #271
      collections are ~6 GB checkpoint blobs (optimizer states under
      ``checkpoint-*/``); the clean ~334 MB adapter still lives at the
      artifact root, so the per-file download yields the clean adapter
      without ever materialising the checkpoint snapshots.

    Always returns a directory containing ``adapter_model.safetensors`` /
    ``adapter_config.json``. Caller verifies usability with
    ``PeftModel.from_pretrained``.
    """
    if adapter_name is None:
        adapter_name = f"marker_{persona}_asst_excluded_medium_seed42"
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if source == "hf_hub":
        from huggingface_hub import snapshot_download

        local = out_root / "hf_hub"
        snapshot_download(
            repo_id=hf_repo_id,
            allow_patterns=[f"adapters/{adapter_name}/*"],
            local_dir=str(local),
        )
        adapter_dir = local / "adapters" / adapter_name
        if not adapter_dir.exists():
            raise FileNotFoundError(
                f"snapshot_download did not produce {adapter_dir}; HF Hub layout changed?"
            )
        return ResolvedAdapter(
            persona=persona,
            source="hf_hub",
            local_dir=adapter_dir,
            artifact_qualified_name=None,
            version=None,
            repo_id=hf_repo_id,
        )

    if source == "wandb":
        import shutil

        import wandb

        api = wandb.Api()
        col_name = f"{wandb_artifact_prefix}/{adapter_name}"
        col = api.artifact_collection(type_name="model", name=col_name)
        versions = list(col.artifacts())
        if not versions:
            raise FileNotFoundError(f"No artifacts in wandb collection {col_name}")

        def _ver_int(v) -> int:
            return int(v.version.lstrip("v"))

        versions_sorted = sorted(versions, key=_ver_int)
        candidates = [v for v in versions_sorted if _ver_int(v) >= min_wandb_version]
        if not candidates:
            raise FileNotFoundError(
                f"No wandb artifact with version >= v{min_wandb_version} for {col_name}; "
                f"available versions: {[v.version for v in versions_sorted]}"
            )

        # Walk versions; for each, download ONLY the two root-level adapter files
        # (``adapter_model.safetensors`` + ``adapter_config.json``) instead of the
        # whole artifact tree. Many of the #271 collections are bloated
        # training-checkpoint blobs (~6 GB: optimizer states + checkpoint
        # snapshots under ``checkpoint-*/``). The clean adapter still lives at
        # the artifact root; selectively downloading those two files yields the
        # ~325 MB clean adapter without ever materialising the checkpoint
        # snapshots — see
        # `.claude/agent-memory/experimenter/feedback_inherited_loras_via_wandb.md`.
        # The previous version's local dir is removed before trying the next
        # version so we never leave half-downloaded blobs on disk.
        adapter_size_cap_bytes = 1_000_000_000  # 1 GB sanity cap on the safetensors itself
        local = out_root / persona
        rejected: list[tuple[str, str]] = []  # (version, reason)
        for cand in candidates:
            if local.exists():
                shutil.rmtree(local)
            local.mkdir(parents=True, exist_ok=True)

            # Look up root-level adapter files in the artifact manifest *without*
            # downloading anything yet. Root-level == no '/' in the manifest path
            # (i.e. NOT inside any ``checkpoint-*/`` subdir).
            files_by_name = {f.name: f for f in cand.files()}
            root_files = {n: f for n, f in files_by_name.items() if "/" not in n}
            if "adapter_model.safetensors" not in root_files:
                rejected.append((cand.version, "no adapter_model.safetensors at artifact root"))
                continue
            if "adapter_config.json" not in root_files:
                rejected.append((cand.version, "no adapter_config.json at artifact root"))
                continue

            # Selective download: only the two root-level adapter files. Use
            # ``ArtifactManifestEntry.download(root=...)`` (per-file API) so we
            # avoid the implicit checkpoint-tree pull of ``cand.download(root)``.
            cand.get_entry("adapter_config.json").download(root=str(local))
            cand.get_entry("adapter_model.safetensors").download(root=str(local))

            adapter_st = local / "adapter_model.safetensors"
            if not adapter_st.exists():
                # Should not happen — get_entry succeeded — but be defensive.
                rejected.append((cand.version, "adapter_model.safetensors missing post-download"))
                continue
            adapter_size = adapter_st.stat().st_size
            if adapter_size >= adapter_size_cap_bytes:
                # The clean Qwen-2.5-7B LoRA adapter is ~334 MB. A larger file at
                # the root would mean either a different (full-weight) checkpoint
                # got committed at the root, or our LoRA rank ballooned — either
                # way we want to crash loudly rather than silently load it.
                rejected.append(
                    (
                        cand.version,
                        f"adapter_model.safetensors size {adapter_size / 1e9:.2f} GB "
                        f">= {adapter_size_cap_bytes / 1e9:.2f} GB cap",
                    )
                )
                continue

            return ResolvedAdapter(
                persona=persona,
                source="wandb",
                local_dir=local,
                artifact_qualified_name=cand.qualified_name,
                version=cand.version,
                repo_id=None,
            )

        raise FileNotFoundError(
            f"No usable wandb artifact for {col_name}; tried {len(candidates)} versions, "
            f"all rejected: {rejected}"
        )

    raise ValueError(f"Unknown source={source!r}; expected 'hf_hub' or 'wandb'")
