# ruff: noqa: RUF001, RUF002  # em-dash + x/marker tokens intentional
"""Task #472 Phase 0.5 — base-model persona centroids + cosine geometry.

Plan §4.3. Distances are a BASE-model geometric property ("where do negatives
sit in base persona space," not "where does training move them"), so centroids
are computed ONCE on the base model over the full ~60-persona bank and reused for
every cell. Uses ``analysis/representation_shift.extract_centroids`` (last-token
hidden state, mean over EVAL_QUESTIONS).

**Cosine methodology — global mean-centering (#504 round-6, restored from
#66/#341).** The bundle stores TWO cosine matrices: ``cos_matrix`` is the raw
(no-centering) matrix kept for backward compatibility with the round 1-5 reads
of #472/#504, and ``cos_matrix_mean_centered`` is the global-mean-centered
matrix recovered from #66's `analyze_100_persona_cosine.py:292-295` pipeline
(centroids minus the per-component mean across the bank, then L2-normalize,
then cosine). Without mean-centering, the cos-to-villain range across a 60-bank
collapses to [~0.92, ~0.99] and the calibration target bands fall on the wrong
side of the saturated geometry — the methodology delta documented in #66's
ρ=0.67–0.87 result. Consumers SHOULD prefer the mean-centered matrix; the raw
matrix is kept solely for resume-from-disk compatibility.

Layers: 10 (headline) + 15 + 20 (robustness; L20 is Persona-Vectors' actual evil
layer, 1-indexed). Distance = 1 − cosine (using the mean-centered cosine).

Output: ``data/issue_472/centroids_L{10,15,20}.pt`` (one bundle per layer) with
``centroids`` (n × d), ``persona_names``, ``cos_matrix`` (raw, legacy alias),
``cos_matrix_mean_centered`` (new — global mean-centered), ``layer``,
``base_model``, ``questions``.

GPU only (one base-model forward pass per persona × 20 questions). A CPU-only
"recompute" path that loads an existing bundle and adds the mean-centered
matrix in place lives in ``scripts/i504_round6_recompute_mean_centered.py``.
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
    ``persona_names`` (ordered), ``cos_matrix`` (n × n, centering="none"; raw —
    the published #472 default), ``cos_matrix_mean_centered`` (n × n, optional
    — global-mean-centered via #66/#341 methodology, written from #504 round-6
    onward and used by #504 callers via the explicit ``centering="global_mean"``
    loader kwarg), ``layer``, ``base_model``, ``questions``.

    The two cosine matrices are co-resident so #472 / #477 replay continues to
    read raw cosines (via the default ``load_cos_matrix(..., centering="none")``
    / ``cos_to_source(..., centering="none")``) while #504 explicitly opts into
    the mean-centered matrix via ``centering="global_mean"``. The shared
    module's default stays raw to keep #472's published numbers reproducible
    bit-for-bit on rerun.

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
        cos_raw = compute_cosine_matrix(c, centering="none")  # (n, n), legacy
        cos_mc = compute_cosine_matrix(c, centering="global_mean")  # (n, n), #66 methodology
        path = _centroids_path(layer, out_dir)
        torch.save(
            {
                "centroids": c,
                "persona_names": persona_names,
                "cos_matrix": cos_raw,  # legacy / backward-compat
                "cos_matrix_mean_centered": cos_mc,
                "layer": layer,
                "base_model": base_model,
                "questions": questions,
            },
            path,
        )
        log.info(
            "Wrote centroids+cos L%d (%d personas, both raw + mean-centered) → %s",
            layer,
            len(persona_names),
            path,
        )
        written[layer] = path
    return written


def load_cos_matrix(
    layer: int,
    out_dir: Path = OUT_DIR,
    *,
    centering: str = "none",
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Load the persona×persona cosine matrix for ``layer`` as a name-keyed dict.

    Returns ``(cos[a][b], persona_names)``.

    Args:
        layer: centroid layer (e.g. 10, 15, 20).
        out_dir: directory holding ``centroids_L{layer}.pt``.
        centering: ``"none"`` (default — published #472 raw cosine; keep this
            default raw so unqualified #472 / #477 replay reproduces the
            promoted numbers bit-for-bit) reads ``cos_matrix``; ``"global_mean"``
            reads the #66/#341 globally-mean-centered ``cos_matrix_mean_centered``
            (round-6 augment, used by #504 callers via explicit opt-in).

    Round-7 fix (binding blocker 2): the default flipped back to ``"none"`` so
    every existing #472 / #477 / #500 caller — ``i472_run_cell.py``,
    ``i472_eval_trajectory.py``, ``i472_phase_base_panel.py``,
    ``i477_reval_confirm.py``, ``analyze.py`` — stays on the published raw
    cosine pipeline without an explicit kwarg. #504 explicitly passes
    ``centering="global_mean"`` at the dispatcher level
    (``scripts/i504_phase_phase05.py``).

    Raises:
        FileNotFoundError: bundle missing.
        KeyError: bundle predates round-6 and the requested ``centering`` key is
            absent (caller can fall back to the other key or re-extract). Only
            the ``"global_mean"`` opt-in is affected by pre-round-6 bundles;
            ``"none"`` is the original schema and always present.
    """
    if centering not in ("global_mean", "none"):
        raise ValueError(f"unsupported centering={centering!r}; use 'global_mean' or 'none'.")
    path = _centroids_path(layer, out_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Centroid bundle for layer {layer} missing at {path}. Run Phase 0.5 "
            f"(centroids.build_centroids) first."
        )
    bundle = torch.load(path, weights_only=False)
    names: list[str] = list(bundle["persona_names"])
    key = "cos_matrix_mean_centered" if centering == "global_mean" else "cos_matrix"
    if key not in bundle:
        # Pre-round-6 bundle missing the new key; fail loud so caller can re-extract or
        # call the round-6 recompute helper.
        raise KeyError(
            f"Centroid bundle at {path} has no {key!r} field (pre-round-6 schema). "
            f"Run scripts/i504_round6_recompute_mean_centered.py to add it in place."
        )
    cos_t: torch.Tensor = bundle[key]
    cos: dict[str, dict[str, float]] = {}
    for i, a in enumerate(names):
        cos[a] = {b: float(cos_t[i, j].item()) for j, b in enumerate(names)}
    return cos, names


def cos_to_source(
    layer: int,
    source: str,
    out_dir: Path = OUT_DIR,
    *,
    centering: str = "none",
) -> dict[str, float]:
    """Return {persona: cos(persona, source)} for every persona in the bank.

    High cosine = near the source. ``select_negatives`` sorts on this to pick
    near/far/spread negatives. Defaults to raw cosine (``centering="none"``)
    to keep the published #472 raw-cosine pipeline reproducible by unqualified
    callers (round-7 fix, binding blocker 2). Pass ``centering="global_mean"``
    to read the #66/#341 globally-mean-centered cosine (round-6 augment,
    used by #504 callers).
    """
    cos, names = load_cos_matrix(layer, out_dir, centering=centering)
    if source not in cos:
        raise KeyError(
            f"Source {source!r} not in centroid bundle (layer {layer}). "
            f"Available: {sorted(names)[:10]}..."
        )
    return dict(cos[source])
