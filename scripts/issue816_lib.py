"""Shared helpers for issue #816 — Persona Vectors' three NON-prediction experiments.

Thin per-issue helpers on TOP of the reused #778 surface (``scripts/issue778_lib``):

- ``fetch_rb`` : download + SHA256-pin the reused #778 ``r_B`` tensors from HF
  (``superkaiba1/explore-persona-space-data`` @
  ``issue778_persona_vectors/analysis_tensors/rb/{trait}.pt``, each ``(28, 3584)``);
  layer 20 (1-indexed) == index 19.
- ``norm_matched_random_dirs`` : the #778 randnorm null draws (covariance-realistic
  N(0,Sigma), diagonal shrinkage lambda=0.1, renormalized to ‖r_B[layer]‖) via
  ``null_battery`` — seeded DETERMINISTICALLY per draw index, saved so the analysis
  is reproducible.
- ``load_eval_conversations`` : the paper's 20 held-out EVALUATION questions per
  trait as single-turn chat conversations (``trait_data_eval/{file}.json``).
- ``read_778_finetune_score`` : read the reused #778 post-ft trait score for a cell
  (``eval_results/issue_778/finetune_{trait}_{family}_misaligned_2.json``) — the
  Exp-4 coef-0 baseline AND the Exp-5 regression y-axis.
- ``assert_778_consumer_paths`` : the plan §12 REQUIRED fail-loud preflight assert.

Everything else (the graded Sonnet judge, activation capture, vLLM engine + reap,
phase logging, sentinel writer, repro metadata) is imported straight from
``issue778_lib`` — no re-implementation.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib

logger = logging.getLogger("issue816.lib")

# Reused #778 r_B directions on the HF DATA repo.
DATA_REPO = "superkaiba1/explore-persona-space-data"
# v2 paths: corrected r_B with coherence gate + unpaired filtering fix
RB_PREFIX = "issue778_persona_vectors/analysis_tensors_v2/rb"
NEUTRAL_COV_PREFIX = "issue778_persona_vectors/analysis_tensors_v2"
# Layer 20 (1-indexed, the paper's steering layer) == 0-indexed block-output index 19.
LAYER_20_IDX = 19
LAYER_20_1IDX = 20

# The paper's released trait-file name per plain-English trait (evil.json /
# sycophantic.json / hallucinating.json) — inherited from issue778_lib.
TRAIT_FILE = lib.TRAIT_FILE
TRAITS = lib.TRAITS
MODEL_NAME = lib.MODEL_NAME
N_LAYERS = lib.N_LAYERS
HIDDEN_DIM = lib.HIDDEN_DIM


def sha256_file(path: Path) -> str:
    """SHA256 of a file (streamed; for pinning the reused r_B tensors)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_rb(trait: str, *, cache_dir: Path):
    """Download the v2 #778 ``r_B[trait]`` tensor from HF and return (tensor, sha256).

    v2 paths: ``issue778_persona_vectors/analysis_tensors_v2/rb/{trait}.pt`` —
    corrected r_B with coherence gate + unpaired filtering fix (plan §4.C Must-Fix B).

    Returns a torch tensor of shape ``(N_LAYERS, HIDDEN_DIM)`` == ``(28, 3584)``
    (0-indexed by block; ``r_B[19]`` == the paper's layer 20) plus the SHA256 of
    the fetched file (pinned in run metadata per plan §12 (f) content identity).
    Fails loud on a shape mismatch or a missing file.

    NOTE: the #778-uploaded r_B tensors are keyed by the PLAIN-ENGLISH trait slug
    (``rb/evil.pt`` / ``rb/sycophancy.pt`` / ``rb/hallucination.pt``), NOT the
    paper's released file name (``sycophantic`` / ``hallucinating``). Verified via
    ``list_repo_files``.
    """
    import torch
    from huggingface_hub import hf_hub_download

    if trait not in TRAITS:
        raise ValueError(f"unknown trait {trait!r}; expected one of {TRAITS}")
    local = hf_hub_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        filename=f"{RB_PREFIX}/{trait}.pt",
        revision="main",
        local_dir=str(cache_dir),
    )
    local_path = Path(local)
    sha = sha256_file(local_path)
    rb = torch.load(local_path, map_location="cpu", weights_only=False)
    if not hasattr(rb, "shape"):
        raise ValueError(f"r_B for {trait} is not a tensor: got {type(rb)}")
    if tuple(rb.shape) != (N_LAYERS, HIDDEN_DIM):
        raise ValueError(
            f"r_B[{trait}] shape {tuple(rb.shape)} != expected ({N_LAYERS}, {HIDDEN_DIM})"
        )
    logger.info("fetched v2 r_B[%s] shape=%s sha256=%s", trait, tuple(rb.shape), sha[:16])
    return rb.float(), sha


def fetch_neutral_cov(trait: str, *, cache_dir: Path):
    """Download the v2 neutral covariance tensor for ``trait`` and return (tensor, sha256).

    Path: ``issue778_persona_vectors/analysis_tensors_v2/neutral_cov_{trait}.pt``
    Shape: ``(N_LAYERS, HIDDEN_DIM, HIDDEN_DIM)`` full covariance matrix, OR
           ``(N_LAYERS, HIDDEN_DIM)`` diagonal approximation — both accepted.
    dtype: float32.

    The neutral covariance is used to construct the honest null family (2)
    independent/neutral-cov draws: N(0, Sigma_neutral_l) renormed to ||r_B[l]||,
    per plan §4.C honest null ladder.
    """
    import torch
    from huggingface_hub import hf_hub_download

    if trait not in TRAITS:
        raise ValueError(f"unknown trait {trait!r}; expected one of {TRAITS}")
    filename = f"{NEUTRAL_COV_PREFIX}/neutral_cov_{trait}.pt"
    local = hf_hub_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        filename=filename,
        revision="main",
        local_dir=str(cache_dir),
    )
    local_path = Path(local)
    # hf_hub_download may nest; find the actual file
    if not local_path.exists():
        candidate = cache_dir / Path(filename).name
        local_path = candidate if candidate.exists() else cache_dir / filename
    sha = sha256_file(local_path)
    t = torch.load(local_path, map_location="cpu", weights_only=True)
    if not isinstance(t, torch.Tensor):
        raise ValueError(f"neutral_cov[{trait}]: expected tensor, got {type(t)}")
    t = t.float()
    # Accept full (N_LAYERS, D, D) or diagonal (N_LAYERS, D)
    ok_full = tuple(t.shape) == (N_LAYERS, HIDDEN_DIM, HIDDEN_DIM)
    ok_diag = tuple(t.shape) == (N_LAYERS, HIDDEN_DIM)
    if not (ok_full or ok_diag):
        raise ValueError(
            f"neutral_cov[{trait}] shape {tuple(t.shape)} is neither "
            f"({N_LAYERS},{HIDDEN_DIM},{HIDDEN_DIM}) nor ({N_LAYERS},{HIDDEN_DIM})"
        )
    logger.info("fetched neutral_cov[%s] shape=%s sha256=%s", trait, tuple(t.shape), sha[:16])
    return t, sha


def norm_matched_random_dirs(
    rb_layer_vec,
    *,
    n_dirs: int,
    pool_acts_layer,
    lam: float = 0.1,
    base_seed: int = 0,
):
    """``n_dirs`` covariance-realistic norm-matched random directions at ONE layer.

    Each direction ~ N(0, Sigma_activations) with diagonal shrinkage ``lam``,
    renormalized to ‖rb_layer_vec‖ (the #778 randnorm null; NOT isotropic). Draw
    ``d`` is seeded DETERMINISTICALLY at ``base_seed + d`` so the analysis is
    reproducible and the SAVED dirs regenerate exactly.

    Args:
        rb_layer_vec: ``(D,)`` r_B at the steering layer (only its NORM is used).
        pool_acts_layer: ``(n_pool, D)`` activation pool at that layer for the
            shrunk covariance (the #778 extraction pos+neg pool at layer 20).
    Returns:
        ``(n_dirs, D)`` float32 numpy array of norm-matched random directions.
    """
    import numpy as np

    from explore_persona_space.analysis import null_battery

    rb_layer_vec = np.asarray(rb_layer_vec, dtype=np.float64)
    target_norm = float(np.linalg.norm(rb_layer_vec))
    chol = null_battery._shrunk_cholesky(np.asarray(pool_acts_layer, dtype=np.float64), lam)
    d = rb_layer_vec.shape[0]
    out = np.empty((n_dirs, d), dtype=np.float64)
    for i in range(n_dirs):
        rng = np.random.default_rng(base_seed + i)
        z = rng.standard_normal(d)
        v = chol @ z
        vn = np.linalg.norm(v)
        out[i] = (v / vn * target_norm) if vn > 0 else v
    return out.astype(np.float32)


def load_eval_conversations(external_root: Path, trait: str) -> list[list[dict]]:
    """The paper's 20 HELD-OUT eval questions for ``trait`` as chat conversations.

    Reuses ``issue778_lib.load_trait_data`` (the released
    ``trait_data_eval/{file}.json`` ``questions`` list) and wraps each as a
    single-turn ``[{"role": "user", "content": q}]`` (persona injection is N/A
    here — the trait is measured on the model's OWN steered/post-ft response to
    the NEUTRAL eval question, per the DV).
    """
    td = lib.load_trait_data(external_root, trait)
    return [[{"role": "user", "content": q}] for q in td.eval_questions]


def load_eval_prompt(external_root: Path, trait: str) -> str:
    """The verbatim paper trait-scoring rubric for ``trait`` (for the graded judge)."""
    return lib.load_trait_data(external_root, trait).eval_prompt


def read_778_finetune_score(
    eval_results_778_root: Path,
    trait: str,
    family: str,
    version: str = "misaligned_2",
) -> dict:
    """Read the reused #778 post-ft trait-score JSON for a cell.

    Consumer-exact path (plan §12): ``finetune_{trait}_{family}_{version}.json``
    under ``eval_results/issue_778/``. Returns the parsed JSON. Fails loud on a
    missing file — the plan's Exp-4 coef-0 baseline + Exp-5 regression y-axis
    depend on it (NO silent fallback).
    """
    path = eval_results_778_root / f"finetune_{trait}_{family}_{version}.json"
    if not path.exists():
        raise FileNotFoundError(
            f"reused #778 finetune score missing: {path} "
            f"(Exp-4 coef-0 / Exp-5 y-axis input — abort rather than silently continue)"
        )
    with open(path) as f:
        return json.load(f)


def assert_778_consumer_paths(
    eval_results_778_root: Path,
    traits: tuple[str, ...] = TRAITS,
    families: tuple[str, ...] = TRAITS,
    version: str = "misaligned_2",
) -> list[str]:
    """Plan §12 REQUIRED preflight assert: every consumed #778 finetune JSON exists.

    Before ANY Exp-4/Exp-5 cell runs, assert every consumer path
    ``finetune_{trait}_{family}_misaligned_2.json`` the plan consumes is present
    in the checkout. Abort fail-loud on a miss (a missing file NEVER falls back to
    a silent default). Returns the list of verified paths (for logging).

    Exp-4 steers trait T while finetuning on the dataset that INDUCES T (the
    misaligned_2 II arm of each of the 3 trait FAMILIES), so the consumed cells
    are the diagonal (trait, family=trait) plus — for Exp-5's 24-dataset y-axis —
    the full family x trait grid the caller passes. The default here checks the
    Exp-4 diagonal (trait==family); Exp-5 passes its own wider set.
    """
    verified: list[str] = []
    missing: list[str] = []
    for trait in traits:
        for family in families:
            path = eval_results_778_root / f"finetune_{trait}_{family}_{version}.json"
            if path.exists():
                verified.append(str(path))
            else:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "reused #778 finetune scores missing (plan §12 preflight assert):\n  "
            + "\n  ".join(missing)
        )
    logger.info("preflight: %d reused #778 finetune JSONs verified present", len(verified))
    return verified
