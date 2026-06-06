# ruff: noqa: RUF001, RUF002, RUF003  # em-dash + × multiplication sign intentional
"""Task #505 §5.7 — base-model persona-vectors centroids at layers {7, 14, 21, 27}.

The #472 centroid bundle on the HF data repo covers L{10, 15, 20} only; the
#505 headline similarity (plan §5.7) is layer 21 (codebase rule
``persona-distance-metrics.md`` legacy default for response-mean-pool
extraction). This module extends ``contrastive_neg_geometry_472.centroids`` to
the new layer set in a single base-model forward pass over the 60-persona bank
× 20 EVAL_QUESTIONS — ~5 GPU-min one-shot (plan §10 row).

Layer indexing convention is **0-indexed transformer blocks** (range 0..27 for
Qwen-2.5-7B's 28 blocks); see plan §5.7 + Assumption 20. ``extract_centroids``
is already layer-parametric (it iterates the ``layers`` arg directly against
``transformer.h`` indices), so the build is a thin wrapper that:

  1. Loads the persona bank from the HF-cached path (mirrors #472).
  2. Calls ``extract_centroids(layers=[7, 14, 21, 27])`` once.
  3. Writes ``centroids_pv_L{layer}.pt`` bundles (one per layer) with the same
     schema as ``centroids_L{10,15,20}.pt``: ``centroids``, ``persona_names``,
     ``cos_matrix`` (``centering="none"``), ``layer``, ``base_model``,
     ``questions``.
  4. Optionally uploads each bundle to the HF data repo under the
     ``issue505_loo_contrastive/geometry/`` subfolder so subsequent runs (and
     the analyze step) can ``hf_hub_download`` them rather than re-running the
     build. Upload is gated by HF_TOKEN presence — local-only when the token is
     missing or ``upload_to_hf=False``.

Mostly mirrors ``contrastive_neg_geometry_472.centroids.build_centroids`` —
kept as a separate module so the #505 layer set + filename prefix
(``centroids_pv_L<layer>.pt``, distinct from #472's ``centroids_L<layer>.pt``)
is single-source-of-truth in #505.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import torch

from explore_persona_space.analysis.representation_shift import (
    compute_cosine_matrix,
    extract_centroids,
)
from explore_persona_space.experiments.leave_one_out_505 import (
    BASE_MODEL,
    HF_DATA_PREFIX,
    HF_DATA_REPO,
    SIMILARITY_LAYERS_TO_BUILD,
)
from explore_persona_space.personas import EVAL_QUESTIONS

log = logging.getLogger("issue_505.build_pv_centroids")


def _upload_bundle_to_hf(local_path: Path, layer: int, *, hf_subfolder: str) -> bool:
    """Upload a single centroid bundle to the HF data repo.

    Returns True on verified upload, False on skipped (no HF_TOKEN) or failure.
    Logs the outcome either way so the caller can decide whether to fail loud.
    """
    token = os.environ.get("HF_TOKEN")
    if not token:
        log.info(
            "[centroids] HF_TOKEN missing; skipping HF upload of %s — bundle stays local-only.",
            local_path.name,
        )
        return False

    from huggingface_hub import HfApi, list_repo_files

    api = HfApi(token=token)
    repo_id = HF_DATA_REPO
    path_in_repo = f"{hf_subfolder}/centroids_pv_L{layer}.pt"
    try:
        api.create_repo(repo_id, repo_type="dataset", private=False, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(local_path),
            repo_id=repo_id,
            path_in_repo=path_in_repo,
            repo_type="dataset",
        )
    except Exception as e:
        log.error(
            "[centroids] HF upload FAILED for %s -> %s/%s: %s",
            local_path.name,
            repo_id,
            path_in_repo,
            e,
        )
        return False

    # Verify via list_repo_files (per CLAUDE.md upload-policy.md mechanics —
    # the hf CLI's silent "0 files" failure mode means we use the Python API).
    try:
        files = list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        log.warning("[centroids] post-upload verify (list_repo_files) failed: %s", e)
        return False
    if path_in_repo not in files:
        log.error(
            "[centroids] post-upload verify FAIL — %s not in repo file list after upload",
            path_in_repo,
        )
        return False
    log.info("[centroids] HF upload OK + verified: %s/%s", repo_id, path_in_repo)
    return True


def _centroid_path(layer: int, out_dir: Path) -> Path:
    """The #505 centroid bundle path; distinct from #472's ``centroids_L<layer>.pt``."""
    return out_dir / f"centroids_pv_L{layer}.pt"


def build_pv_centroids(
    persona_bank: dict[str, str],
    *,
    layers: tuple[int, ...] = SIMILARITY_LAYERS_TO_BUILD,
    questions: list[str] | None = None,
    base_model: str = BASE_MODEL,
    out_dir: Path,
    device: str = "cuda:0",
    skip_existing: bool = True,
    upload_to_hf: bool = True,
    hf_subfolder: str = f"{HF_DATA_PREFIX}/geometry",
) -> dict[int, Path]:
    """Extract base-model centroids at the persona-vectors layer set + write per-layer bundles.

    Layers are 0-indexed transformer blocks; see plan §5.7 + Assumption 20.

    Args:
        persona_bank: name -> system prompt for the full ~60-persona bank.
        layers: 0-indexed transformer blocks to extract. Default is the §5.7
            headline + robustness set {7, 14, 21, 27}; the L10 layer is omitted
            because the #472 bundle on the HF data repo already covers it.
        questions: eval questions to mean-pool over (default ``EVAL_QUESTIONS``,
            20 items). Mirrors #472's centroid recipe so the layer-21 similarity
            is mechanically the same recipe as the layer-10 fallback.
        base_model: HF model id.
        out_dir: directory where ``centroids_pv_L{layer}.pt`` lands.
        device: device string for the forward pass.
        skip_existing: if True, skip layers whose bundle file already exists on
            disk (idempotent re-runs).
        upload_to_hf: if True AND HF_TOKEN is set, upload each freshly-built
            bundle to ``HF_DATA_REPO`` under ``hf_subfolder/`` so subsequent
            runs can ``hf_hub_download`` instead of re-running the build.
            local-only when ``upload_to_hf=False`` or HF_TOKEN is missing.
        hf_subfolder: destination path inside the HF data repo. Default lands
            the bundles next to the #505 cell artifacts.

    Returns:
        dict layer -> written path. Includes pre-existing bundles when
        ``skip_existing=True``.
    """
    if questions is None:
        questions = list(EVAL_QUESTIONS)
    out_dir.mkdir(parents=True, exist_ok=True)

    if skip_existing:
        missing = [layer for layer in layers if not _centroid_path(layer, out_dir).exists()]
        existing = {
            layer: _centroid_path(layer, out_dir) for layer in layers if layer not in missing
        }
        if existing:
            log.info(
                "[centroids] skipping %d already-built layers: %s", len(existing), sorted(existing)
            )
        if not missing:
            log.info("[centroids] all %d layers already on disk; nothing to build.", len(layers))
            return existing
        build_layers = tuple(missing)
    else:
        existing = {}
        build_layers = tuple(layers)

    log.info(
        "[centroids] Extracting %d personas × %d questions × %d layers (%s) → %s",
        len(persona_bank),
        len(questions),
        len(build_layers),
        build_layers,
        out_dir,
    )
    centroids, persona_names = extract_centroids(
        model_path=base_model,
        personas=persona_bank,
        questions=questions,
        layers=list(build_layers),
        device=device,
    )

    written = dict(existing)
    for layer in build_layers:
        c = centroids[layer]  # (n, d) float32 on CPU
        cos = compute_cosine_matrix(c, centering="none")  # (n, n)
        path = _centroid_path(layer, out_dir)
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
        log.info(
            "[centroids] wrote L%d (%d personas, cos %s) → %s",
            layer,
            len(persona_names),
            cos.shape,
            path,
        )
        written[layer] = path
        if upload_to_hf:
            _upload_bundle_to_hf(path, layer, hf_subfolder=hf_subfolder)
    return written


def load_pv_cos(layer: int, out_dir: Path) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Load a #505 centroid bundle for a layer as a name-keyed cosine dict.

    Returns ``(cos[a][b], persona_names)`` where ``cos[a][b]`` is the
    centering="none" cosine between persona a and b centroids (same recipe as
    #472's bundle).

    Raises FileNotFoundError if the bundle is absent — call ``build_pv_centroids``
    first or pull it from HF.
    """
    path = _centroid_path(layer, out_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"Persona-vectors centroid bundle for layer {layer} missing at {path}. "
            f"Run build_pv_centroids first or fetch from the HF data repo."
        )
    bundle = torch.load(path, weights_only=False)
    names: list[str] = list(bundle["persona_names"])
    cos_t = bundle["cos_matrix"]
    cos: dict[str, dict[str, float]] = {}
    for i, a in enumerate(names):
        cos[a] = {b: float(cos_t[i, j].item()) for j, b in enumerate(names)}
    return cos, names
