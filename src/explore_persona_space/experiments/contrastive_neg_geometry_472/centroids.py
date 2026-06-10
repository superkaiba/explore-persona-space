# ruff: noqa: RUF001, RUF002  # em-dash + x/marker tokens intentional
"""Task #472 Phase 0.5 — base-model persona centroids + cosine geometry.

Plan §4.3. Distances are a BASE-model geometric property ("where do negatives
sit in base persona space," not "where does training move them"), so centroids
are computed ONCE on the base model over the full ~60-persona bank and reused for
every cell. Uses ``analysis/representation_shift.extract_centroids`` (last-token
hidden state, mean over EVAL_QUESTIONS) + ``compute_cosine_matrix(centering=
"none")`` (NOT the static ASSISTANT_COSINES dict).

Layers: 10 (headline) + 15 + 20 (robustness; L20 is Persona-Vectors' actual evil
layer, 1-indexed). Distance = 1 − cosine.

Output: ``data/issue_472/centroids_L{10,15,20}.pt`` (one bundle per layer) plus a
derived ``cos_matrix`` per layer keyed by persona name. The derived per-probe
covariates (d_source, d_nearest_neg, d_nearest_neg_nd) are computed in
``select_negatives`` / ``analyze`` from these matrices.

GPU only (one base-model forward pass per persona × 20 questions).
"""

from __future__ import annotations

import logging
from pathlib import Path

import torch

from explore_persona_space.analysis.representation_shift import (
    compute_cosine_matrix,
    extract_centroids,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
    BASE_MODEL,
    CENTROID_LAYERS,
)
from explore_persona_space.personas import EVAL_QUESTIONS

log = logging.getLogger("issue_472.centroids")

OUT_DIR = Path("data/issue_472")


def _centroids_path(layer: int, out_dir: Path = OUT_DIR) -> Path:
    return out_dir / f"centroids_L{layer}.pt"


def build_centroids(
    persona_bank: dict[str, str],
    *,
    layers: tuple[int, ...] = CENTROID_LAYERS,
    questions: list[str] | None = None,
    base_model: str = BASE_MODEL,
    out_dir: Path = OUT_DIR,
    device: str = "cuda:0",
) -> dict[int, Path]:
    """Extract base-model centroids over the bank and write per-layer .pt bundles.

    Each bundle holds: ``centroids`` (n_personas × hidden_dim, float32),
    ``persona_names`` (ordered), ``cos_matrix`` (n × n, centering="none"),
    ``layer``, ``base_model``.

    Args:
        persona_bank: name -> system prompt for the full ~60-persona bank.
        layers: centroid layers to extract (default 10/15/20).
        questions: eval questions to average over (default EVAL_QUESTIONS, 20).
        base_model: HF model id.
        out_dir: directory for centroids_L{layer}.pt.
        device: device string.

    Returns:
        dict layer -> written path.
    """
    if questions is None:
        questions = list(EVAL_QUESTIONS)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info(
        "Extracting base centroids: %d personas × %d questions, layers=%s",
        len(persona_bank),
        len(questions),
        layers,
    )
    centroids, persona_names = extract_centroids(
        model_path=base_model,
        personas=persona_bank,
        questions=questions,
        layers=list(layers),
        device=device,
    )

    written: dict[int, Path] = {}
    for layer in layers:
        c = centroids[layer]  # (n, d) float32 on CPU
        cos = compute_cosine_matrix(c, centering="none")  # (n, n)
        path = _centroids_path(layer, out_dir)
        torch.save(
            {
                "centroids": c,
                "persona_names": persona_names,
                "cos_matrix": cos,
                "layer": layer,
                "base_model": base_model,
                "questions": questions,
            },
            path,
        )
        log.info("Wrote centroids+cos L%d (%d personas) → %s", layer, len(persona_names), path)
        written[layer] = path
    return written


def load_cos_matrix(
    layer: int, out_dir: Path = OUT_DIR
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Load the persona×persona cosine matrix for ``layer`` as a name-keyed dict.

    Returns ``(cos[a][b], persona_names)`` where ``cos[a][b]`` is the
    centering="none" cosine between persona a and b centroids.

    Raises FileNotFoundError if the bundle is absent.
    """
    path = _centroids_path(layer, out_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Centroid bundle for layer {layer} missing at {path}. Run Phase 0.5 "
            f"(centroids.build_centroids) first."
        )
    bundle = torch.load(path, weights_only=False)
    names: list[str] = list(bundle["persona_names"])
    cos_t: torch.Tensor = bundle["cos_matrix"]
    cos: dict[str, dict[str, float]] = {}
    for i, a in enumerate(names):
        cos[a] = {b: float(cos_t[i, j].item()) for j, b in enumerate(names)}
    return cos, names


def cos_to_source(
    layer: int,
    source: str,
    out_dir: Path = OUT_DIR,
) -> dict[str, float]:
    """Return {persona: cos(persona, source)} for every persona in the bank.

    High cosine = near the source. ``select_negatives`` sorts on this to pick
    near/far/spread negatives.
    """
    cos, names = load_cos_matrix(layer, out_dir)
    if source not in cos:
        raise KeyError(
            f"Source {source!r} not in centroid bundle (layer {layer}). "
            f"Available: {sorted(names)[:10]}..."
        )
    return dict(cos[source])
