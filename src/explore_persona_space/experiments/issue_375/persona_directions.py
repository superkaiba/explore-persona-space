"""Persona direction extraction for issue #375.

Wraps :func:`explore_persona_space.analysis.representation_shift.extract_centroids`
to extract L20 persona directions on the *base* Qwen-2.5-7B-Instruct model
using the Chen et al. 2025 persona-vector recipe:

    v_P = normalize(mean_last_token[L20, persona-prompt] -
                    mean_last_token[L20, assistant-prompt])

over the 20 v3 ``EVAL_QUESTIONS`` probe set.

Hard cosine-spread gate (plan §4.3): any pairwise cosine > 0.95 between the
three source persona directions raises ``RuntimeError`` — the directions are
collinear and the per-persona top-K pools would be near-identical.

Caches the unit-norm directions + the raw centroids tensor to
``data/issue_375/persona_directions_L20.pt`` so the eval loop never re-extracts.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import torch

from explore_persona_space.analysis.representation_shift import extract_centroids
from explore_persona_space.personas import EVAL_QUESTIONS

log = logging.getLogger(__name__)

SOURCE_PERSONAS: tuple[str, ...] = ("software_engineer", "librarian", "villain")
NEUTRAL_NAME = "assistant"


@dataclass
class PersonaDirections:
    """Container for the L20 unit-norm persona directions and raw centroids.

    Attributes:
        layer: Layer index the directions were extracted at (always 20 for
            this experiment).
        directions: ``{persona_name: Tensor(hidden_dim,)}`` — unit-norm
            ``v_P = (centroid_P - centroid_assistant) / ||...||``.
        centroids: ``Tensor(n_personas, hidden_dim)`` — raw centroids
            (persona-prompt mean last-token activation) at the layer.
        persona_names: Ordering of rows in ``centroids``.
        pairwise_cos: ``{(p, q): cos(v_p, v_q)}`` for every unordered pair of
            source personas — for the §4.3 sanity diagnostic.
        recipe: Human-readable description for the result JSON.
    """

    layer: int
    directions: dict[str, torch.Tensor]
    centroids: torch.Tensor
    persona_names: list[str]
    pairwise_cos: dict[tuple[str, str], float] = field(default_factory=dict)
    recipe: str = (
        "Chen et al. 2025 persona-vector: normalize(mean_last_token[L=layer, persona-prompt] - "
        "mean_last_token[L=layer, assistant-prompt]) over the 20 v3 EVAL_QUESTIONS."
    )


def _pairwise_cosines(
    directions: Mapping[str, torch.Tensor],
) -> dict[tuple[str, str], float]:
    """Return ``{(p, q): cos(v_p, v_q)}`` for every unordered pair p != q."""
    names = sorted(directions.keys())
    out: dict[tuple[str, str], float] = {}
    for i, p in enumerate(names):
        for q in names[i + 1 :]:
            v_p = directions[p]
            v_q = directions[q]
            cos = torch.dot(v_p, v_q).item()
            out[(p, q)] = float(cos)
    return out


def extract_persona_directions(
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
    layer: int = 20,
    personas: Mapping[str, str] | None = None,
    questions: list[str] | None = None,
    device: str = "cuda:0",
    cosine_pairwise_hard: float = 0.95,
    cosine_pairwise_warn: float = 0.90,
) -> PersonaDirections:
    """Extract unit-norm persona directions at ``layer`` on ``base_model``.

    Args:
        base_model: HF Hub path. Persona directions are a property of the
            base model's representations, not adapted models.
        layer: Layer index to extract last-token activations at.
        personas: ``{name: system_prompt}``. Must include the 3 source personas
            (software_engineer, librarian, villain) and an ``assistant``
            neutral reference. Defaults to the canonical v3 prompts.
        questions: Probe questions for the centroid mean. Defaults to the 20
            v3 ``EVAL_QUESTIONS``.
        device: cuda device string passed through to ``extract_centroids``.
        cosine_pairwise_hard: If any pairwise cos between source-persona
            directions exceeds this threshold, raise ``RuntimeError``.
        cosine_pairwise_warn: Warn (don't fail) above this threshold.

    Returns:
        Populated :class:`PersonaDirections`.

    Raises:
        RuntimeError: when the cosine-spread hard gate fires.
    """
    if personas is None:
        personas = {
            "software_engineer": "You are a software engineer who builds web applications.",
            "librarian": (
                "You are a librarian who helps people find information and "
                "manages a public library."
            ),
            "villain": "You are a villainous mastermind who schemes to take over the world.",
            NEUTRAL_NAME: "You are a helpful assistant.",
        }
    if questions is None:
        questions = EVAL_QUESTIONS

    for required in (*SOURCE_PERSONAS, NEUTRAL_NAME):
        if required not in personas:
            raise ValueError(
                f"persona_directions: required persona {required!r} missing from `personas`"
            )

    log.info(
        "extracting persona centroids: base=%s layer=%d personas=%s n_questions=%d",
        base_model,
        layer,
        list(personas.keys()),
        len(questions),
    )

    centroids_dict, persona_names = extract_centroids(
        model_path=base_model,
        personas=dict(personas),
        questions=list(questions),
        layers=[layer],
        device=device,
    )
    centroids = centroids_dict[layer]  # Tensor(n_personas, hidden_dim)

    asst_idx = persona_names.index(NEUTRAL_NAME)
    asst_centroid = centroids[asst_idx]

    directions: dict[str, torch.Tensor] = {}
    for p in SOURCE_PERSONAS:
        p_idx = persona_names.index(p)
        v = centroids[p_idx] - asst_centroid
        norm = v.norm()
        if not torch.isfinite(norm) or norm < 1e-8:
            raise RuntimeError(
                f"persona_directions: degenerate direction for persona={p!r} at layer={layer} "
                f"(norm={norm.item():.3e}). Centroid extraction produced a near-zero vector."
            )
        directions[p] = v / norm

    pairwise = _pairwise_cosines(directions)
    log.info("pairwise persona-direction cosines at L%d: %s", layer, pairwise)

    max_cos = max(pairwise.values()) if pairwise else 0.0
    if max_cos > cosine_pairwise_hard:
        raise RuntimeError(
            f"persona_directions: cosine spread HARD GATE fired — "
            f"max pairwise cos = {max_cos:.4f} > {cosine_pairwise_hard:.4f}. "
            f"Source persona directions are collinear at L{layer}; the per-persona "
            f"top-K pools would be near-identical. Pairwise cos: {pairwise}"
        )
    if max_cos > cosine_pairwise_warn:
        log.warning(
            "persona_directions: pairwise cos %.4f exceeds WARN threshold %.4f "
            "(hard gate at %.4f). Inspect pools manually.",
            max_cos,
            cosine_pairwise_warn,
            cosine_pairwise_hard,
        )

    return PersonaDirections(
        layer=layer,
        directions=directions,
        centroids=centroids.cpu(),
        persona_names=list(persona_names),
        pairwise_cos=pairwise,
    )


def save_persona_directions(directions: PersonaDirections, path: str | Path) -> None:
    """Save persona directions + centroids to a single ``.pt`` file.

    The on-disk schema is::

        {
            "layer": int,
            "directions": {persona: Tensor(hidden_dim)},
            "centroids": Tensor(n_personas, hidden_dim),
            "persona_names": list[str],
            "pairwise_cos": {f"{p}|{q}": float},
            "recipe": str,
        }
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "layer": directions.layer,
        "directions": {k: v.cpu() for k, v in directions.directions.items()},
        "centroids": directions.centroids.cpu(),
        "persona_names": list(directions.persona_names),
        "pairwise_cos": {f"{p}|{q}": v for (p, q), v in directions.pairwise_cos.items()},
        "recipe": directions.recipe,
    }
    torch.save(payload, path)
    log.info("saved persona directions to %s", path)


def load_persona_directions(path: str | Path) -> PersonaDirections:
    """Load persona directions from disk (inverse of :func:`save_persona_directions`)."""
    payload = torch.load(path, weights_only=False, map_location="cpu")
    pairwise = {}
    for key, val in payload.get("pairwise_cos", {}).items():
        p, q = key.split("|", 1)
        pairwise[(p, q)] = float(val)
    return PersonaDirections(
        layer=int(payload["layer"]),
        directions={k: v.float() for k, v in payload["directions"].items()},
        centroids=payload["centroids"].float(),
        persona_names=list(payload["persona_names"]),
        pairwise_cos=pairwise,
        recipe=payload.get("recipe", PersonaDirections.recipe),
    )
