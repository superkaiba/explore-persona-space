"""Shared helpers for issue #816 — Persona Vectors' three NON-prediction experiments.

Thin per-issue helpers on TOP of the reused #778 surface (``scripts/issue778_lib``):

- ``fetch_rb`` : download + SHA256-pin the reused #778 v2 ``r_B`` tensors from HF
  (``superkaiba1/explore-persona-space-data`` @
  ``issue778_persona_vectors/analysis_tensors_v2/rb_v2/{trait}.pt``, each
  ``(28, 3584)``); layer 20 (1-indexed) == index 19.
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
# v2 paths: corrected r_B with coherence gate + paired filtering fix. #778's v2
# upload contract (issue778_v2_upload.py, issue-778-v2rerun branch) writes the
# r_B tensors under rb_v2/ and NO precomputed neutral_cov tensors — the neutral
# covariance derives from neutral/neutral_response_avg.pt (see fetch_neutral_cov).
RB_PREFIX = "issue778_persona_vectors/analysis_tensors_v2/rb_v2"
NEUTRAL_RESP_AVG_FILE = (
    "issue778_persona_vectors/analysis_tensors_v2/neutral/neutral_response_avg.pt"
)
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

    v2 paths: ``issue778_persona_vectors/analysis_tensors_v2/rb_v2/{trait}.pt`` —
    corrected r_B with coherence gate + paired filtering fix (plan §4.C Must-Fix B).
    A trait can be ABSENT under rb_v2/ (#778's K1 gate: < 5 kept judge pairs ⇒ no
    r_B v2 written) — hf_hub_download then raises, which is the correct fail-loud
    behavior (the Phase-0 poll exits 4 on that case before any cell consumes).

    Returns a torch tensor of shape ``(N_LAYERS, HIDDEN_DIM)`` == ``(28, 3584)``
    (0-indexed by block; ``r_B[19]`` == the paper's layer 20) plus the SHA256 of
    the fetched file (pinned in run metadata per plan §12 (f) content identity).
    Fails loud on a shape mismatch or a missing file.

    NOTE: the #778-uploaded r_B tensors are keyed by the PLAIN-ENGLISH trait slug
    (``rb_v2/evil.pt`` / ``rb_v2/sycophancy.pt`` / ``rb_v2/hallucination.pt``), NOT
    the paper's released file name (``sycophantic`` / ``hallucinating``) — per
    #778's writer (issue778_extract.py judge_and_build_v2).
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


# The neutral corpus is trait-independent, so the derived covariance is computed
# once per process and shared across the three per-trait calls.
_NEUTRAL_COV_CACHE: dict[str, tuple] = {}


def fetch_neutral_cov(trait: str, *, cache_dir: Path):
    """Derive the v2 neutral covariance from #778's neutral corpus activations.

    #778 v2 uploads NO precomputed neutral_cov tensors: the honest neutral-cov
    null derives from ``neutral/neutral_response_avg.pt`` — shape
    ``(n, N_LAYERS, HIDDEN_DIM)`` fp32, one response-averaged activation row per
    neutral UltraChat prompt (issue778_neutral_capture.py). Per layer:
    ``Sigma_emp = np.cov(acts64, rowvar=False)`` then shrunk
    ``Sigma = (1-lam)*Sigma_emp + lam*diag(Sigma_emp)`` with
    ``lam = null_battery.PRIMARY_LAMBDA`` — the SAME construction #778's honest
    null ladder applies via ``null_battery._shrunk_cholesky``, so the null
    construct matches across the two tasks. Shrinkage is applied exactly ONCE
    here; downstream consumers (screening family 2, steering e2/e4_neutral_cov)
    Cholesky the returned matrix directly with only a tiny PD jitter.

    ``trait`` is accepted for call-site compatibility but ignored for the data —
    all traits share ONE covariance (computed once, cached per process).

    Returns ``(cov, sha256)``: cov ``(N_LAYERS, HIDDEN_DIM, HIDDEN_DIM)`` float32
    torch tensor; sha256 of the SOURCE neutral_response_avg.pt file (content-
    identity pin per plan §12 (f)).
    """
    import numpy as np
    import torch
    from huggingface_hub import hf_hub_download

    from explore_persona_space.analysis.null_battery import PRIMARY_LAMBDA

    if trait not in TRAITS:
        raise ValueError(f"unknown trait {trait!r}; expected one of {TRAITS}")
    if "cov" in _NEUTRAL_COV_CACHE:
        return _NEUTRAL_COV_CACHE["cov"]
    local = hf_hub_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        filename=NEUTRAL_RESP_AVG_FILE,
        revision="main",
        local_dir=str(cache_dir),
    )
    local_path = Path(local)
    sha = sha256_file(local_path)
    acts = torch.load(local_path, map_location="cpu", weights_only=True)
    if not isinstance(acts, torch.Tensor):
        raise ValueError(f"neutral_response_avg: expected tensor, got {type(acts)}")
    if acts.ndim != 3 or tuple(acts.shape[1:]) != (N_LAYERS, HIDDEN_DIM):
        raise ValueError(
            f"neutral_response_avg shape {tuple(acts.shape)} != (n, {N_LAYERS}, {HIDDEN_DIM})"
        )
    acts_np = acts.numpy()
    cov = torch.empty((N_LAYERS, HIDDEN_DIM, HIDDEN_DIM), dtype=torch.float32)
    for layer in range(N_LAYERS):
        acts64 = acts_np[:, layer, :].astype(np.float64)
        cov_emp = np.cov(acts64, rowvar=False)  # (D, D)
        shrunk = (1.0 - PRIMARY_LAMBDA) * cov_emp + PRIMARY_LAMBDA * np.diag(np.diag(cov_emp))
        cov[layer] = torch.from_numpy(shrunk).float()
    logger.info(
        "derived neutral_cov from %s (n=%d) shape=%s lam=%.3f source_sha=%s",
        NEUTRAL_RESP_AVG_FILE,
        acts.shape[0],
        tuple(cov.shape),
        PRIMARY_LAMBDA,
        sha[:16],
    )
    _NEUTRAL_COV_CACHE["cov"] = (cov, sha)
    return cov, sha


def norm_matched_random_dirs(
    rb_layer_vec,
    *,
    n_dirs: int,
    pool_acts_layer,
    lam: float = 0.1,
    base_seed: int = 0,
):
    """CONTAMINATED-REFERENCE-ONLY: uses pos+neg pooled covariance whose top PC ~ r_B.

    This function samples directions from the pooled pos+neg activation covariance,
    which is contaminated (top PC cos~0.996 with r_B) and therefore INVALID as a null.
    It is retained ONLY as the reference for Family 8 in screening.py (the contaminated
    pool-based null, kept to show why it fails).

    DO NOT use this in production cells for Exp-2 or Exp-4. Use the honest arms:
      - ``e2_isotropic`` / ``e4_isotropic``: N(0, I·σ²) renormed
      - ``e2_neutral_cov`` / ``e4_neutral_cov``: Cholesky from neutral_cov renormed

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
