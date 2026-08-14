"""Issue #2254 driver: the persona vector's pre-image as a causal context-vector
steering direction.

Phase-dispatch driver (``--phase``; PHASES registry), mirroring
``scripts/issue2220_readwrite.py``'s argparse / resume / sentinel machinery.

Phases (plan §4):
  stage_inputs         e1-asset staging (sha-pinned) + A8 disjointness gate
                       extended with the pass-B LMSYS-prompt overlap check.
  fit_maps             per-layer ridge maps on the #779 pass-B bundle (28 real
                       + 28 shuffled float64 SVD fits, ProcessPool across
                       layers), pre-image directions, HALT-class parity +
                       frame-fold gates, mapping-baselines report, HF upload.
  capture_directions   direction 3 (context-extracted diff-of-means over the
                       200 extraction contexts) + full direction bank assembly
                       + Result 0 geometry cosines + bank upload.  GPU.
  norm_probe           rho_l at all 28 layers (per-behavior + pooled medians)
                       with the #2220 shared-layer parity assert, plus the
                       1-cell production-shape timing pilot (gate-1 input). GPU.
  baseline_ceiling     alpha=0 baselines + donor-swap ceilings, decisive grain
                       (20q x 5 draws x seeds {42,43}; ceiling 100 ctx x 1). GPU.
  localize             wave-1 dose x direction x position x layer-config grid
                       (385 cells/behavior incl. alpha0; 10q x 3 draws x s42),
                       per-cell checkpoints + shard-safe resume.  GPU.
  decisive             wave-2 confirmation at the judge-selected operating
                       points (20q x 5 draws x seeds {42,43}); pod-B rho seam
                       re-asserted vs the staged rho_by_layer.json.  GPU.
  patch                calibration captures (mu_pos/mu_neut projections) +
                       projection-patch / directional-ablation cells.  GPU.
  build_pools          margin answer pools: evil+sycophancy staged verbatim
                       from #2220; hallucination REGENERATED from judged
                       localize completions (planned step, same instrument).
  margin               teacher-forced fixed +/- pool margin at the decisive
                       single-breadth operating points (#2220 rig, batch 4). GPU.
  judge_reduce         off-pod: pilot-gated Batch-API judging + wave reduce
                       (dose_response / operating_points / gates / verdicts /
                       patch_vs_ceiling; selection-symmetric null bands).
  figures              off-pod: hero + diagnostic figures (issue2254_figures).

Reuse provenance (plan §10 "Reused code" row):
  - ``ridge_fit_matrix`` is copied VERBATIM from
    ``eval_results/issue_779/pinv_topk_contexts/pinv_topk_contexts.py`` L92-120
    (itself verbatim from ``pinv_direction_read.py``, verified there to
    reproduce ``fit_h.ridge_fit_predict`` to machine precision).
  - the pre-image construction (SVD of M = W.T, k* = #{i: s_i^2 >= lambda}
    counted on the FIT's standardized-X singular values, truncated pinv
    ``V_k diag(1/s_k) U_k^T r_B``) mirrors
    ``eval_results/issue_779/pinv_direction_read/pinv_direction_read.py``
    L160-207 — the producer of the committed parity reference.
  - ``random_direction`` is copied VERBATIM from ``issue2220_readwrite.py``
    L345-361; ``_assert_eval_bank_disjoint`` / ``_norm_question`` are IMPORTED
    from that module (behavior-generic; our corpus set threads through the
    ``corpus_texts=`` parameter).
  - generation/capture reuse: ``issue1415.steering.generate_batch`` /
    ``capture_vectors`` / ``DeltaHook`` (signatures read at steering.py
    L382/L483/L144).

Sentinels: ``/workspace/logs/issue-2254-<phase>.json`` (pod-observed; the VM
poller drains them — pod-side code NEVER shells to task.py).

Content hygiene: eval/extraction question text and LMSYS prompt text are passed
to the model/judge and set-compared, never logged (digests only).
"""

from __future__ import annotations

# hf_transfer acceleration must be in the env BEFORE any transitive
# huggingface_hub import (constants freeze at import time) — first lines.
import os

os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

# load_dotenv BEFORE any numpy/torch import (thread-cap + credential
# setdefaults freeze at BLAS/torch import; orchestrate.env, never bare dotenv).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402  (after load_dotenv so BLAS thread caps apply)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stdout
)
logger = logging.getLogger("issue2254")


def _ensure_repo_root_on_syspath() -> None:
    """Put the repo root on sys.path so `import scripts.<mod>` resolves (#823).

    In script mode sys.path[0] is the script's own dir (`scripts/`), so the
    `scripts` PACKAGE (needed for the issue2220_readwrite / issue779_common
    reuse) is unimportable without the repo root on the path. Idempotent;
    asserts a repo sentinel so a wrong parent index fails loud.
    """
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "pyproject.toml").exists(), f"repo-root sentinel missing at {repo_root}"
    p = str(repo_root)
    if p not in sys.path:
        sys.path.insert(0, p)


# ---------------------------------------------------------------------------
# pins (plan §4/§9/§10)
# ---------------------------------------------------------------------------
from explore_persona_space.experiments.issue_1739.constants import (  # noqa: E402
    HF_DATA_REPO,
    HIDDEN_DIM,
    MODEL_NAME,
)

ISSUE = 2254
BEHAVIORS = ("evil", "sycophancy", "hallucination")
N_LAYERS = 28
ALL_LAYERS = tuple(range(N_LAYERS))

# #779 pin (pass-B bundle + r_B bank + read-direction lineage; plan §10).
HF_REV = "037fcbb210bc52c459959b0746cc268fe08bae96"
PASS_B_FILE = "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"

# HF destinations (plan §10). Smoke uploads divert to the smoke/ sub-prefix so
# smoke artifacts (production-identical names) never overwrite canonical ones.
HF_PREFIX = "issue2254_preimage"
_SMOKE_UPLOAD_SUBPREFIX = False


def _hf_prefix() -> str:
    """Data-repo prefix for all uploads; `<HF_PREFIX>/smoke` under --smoke."""
    return f"{HF_PREFIX}/smoke" if _SMOKE_UPLOAD_SUBPREFIX else HF_PREFIX


# e1 extraction-asset sha256 pins (plan §10 "Reused: extraction/eval banks"
# row; VERIFIED against the VM-local `data/issue_779/artifacts/*.json` copies
# on 2026-08-12 — the dispatch step uploads exactly these bytes to
# `issue2254_preimage/inputs/e1_assets/`). Evil is code-resident
# (paper-verbatim `issue779_common.EVIL_ARTIFACTS`), no file to pin.
E1_ASSET_SHA256 = {
    "sycophancy": "998f4643666df9e5d2ed3def4275b839b3d40ea426079abf865279f034b2f212",
    "hallucination": "053afaa273371d3806c744d5fb0aaf6fe3a983616f2052cd4d723ec14f843d5e",
}
E1_INPUT_PREFIX = f"{HF_PREFIX}/inputs/e1_assets"  # canonical (non-smoke) staging source
N_EVAL_QUESTIONS = 20  # persona-vectors disjoint eval set size (plan §4.4)
N_EXTRACTION_QUESTIONS = 20
N_INSTRUCTION_PAIRS = 5

# Frozen-layer parity targets (plan §4.1 HALT gate). Values are the #1615
# pre-registered k* (KSTAR_PREREG in pinv_topk_contexts.py L84) and the frozen
# read-out layers; lambda + recon R2 are read from the COMMITTED
# pinv_direction_read.json at runtime (these constants cross-check that the
# committed file is the intended one).
FROZEN_LAYER = {"evil": 14, "sycophancy": 26, "hallucination": 17}
KSTAR_PREREG = {"evil": 1433, "sycophancy": 1321, "hallucination": 1565}
PARITY_JSON_REL = "eval_results/issue_779/pinv_direction_read/pinv_direction_read.json"
PARITY_CONE = "eval_results/issue_779/pinv_direction_read"
# |recon R2 refit - committed| tolerance: float64 SVD cross-BLAS jitter is
# ~1e-10; real drift (bundle rows / recipe change) moves R2 at >=1e-2. 1e-4
# keeps 6 orders of margin on both sides.
PARITY_R2_ATOL = 1e-4

# Frame-fold HALT threshold (plan §4.1): the fold is exact algebra, so the
# cosine is 1 up to float error; 0.999 catches any wrong-frame construction.
FRAME_FOLD_MIN_COS = 0.999

# Held-out mapping-baselines split (plan §4.1 mapping-baselines duty; a
# fit-quality report, not a headline — pointwise 90/10 per the OOD-folds
# declaration in plan §6). Split seed is a recorded convention (issue number).
HELDOUT_FRAC = 0.10
HELDOUT_SPLIT_SEED = 2254
KNN_KS = (1, 5, 10)  # k=10 headline (chance = 10/n_heldout stated in-report)

# Shuffled-map control (plan §4.1 row 5): Y rows permuted with seed 2254 (one
# permutation, shared across layers, drawn in the parent).
SEED_SHUFFLE = 2254
# Random control (plan §4.1 row 4): the #2220 construction — per-layer seed =
# base + layer, 3 seeded unit draws averaged (random_direction below draws
# default_rng(seed*1000 + i) internally; plan §10's "2254+i" names this
# 3-draws-from-2254 lineage).
SEED_RANDOM_BASE = 2254
N_RANDOM_SEEDS = 3

# norm_probe parity vs #2220 (plan §4.2): same model, same eval banks, same
# single-row unpadded forward geometry (matched batch geometry — gotchas.md
# "Gate reads need matched batch geometry"), so agreement is expected to
# ~float level; 0.5% relative catches every real failure mode (wrong position
# / wrong layer / wrong bank are >10% effects) while absorbing bf16 kernel
# jitter across transformers/driver versions (bf16 GPU parity-tolerance
# family, gotchas.md; artifact-reuse.md § gate calibration).
#
# Tolerance split (production HALT diagnosis, #2254 events v20, 2026-08-13):
# the flat 5e-3 stays BINDING for behaviors whose parity inputs are
# code-resident on BOTH sides (evil — the paper-verbatim #779 prompts live in
# scripts/issue779_common.py, so #2220 and #2254 forwarded identical text:
# the kernel/recipe canary). The two BANK-LOADED behaviors (hallucination,
# sycophancy) get RHO_PARITY_RTOL_BANKED because the REFERENCE side's bank
# provenance is not pinned: #2220's events record its eval banks as
# data/issue_779/artifacts/<trait>.json — an UNTRACKED pod-local cache with a
# standing Sonnet-regeneration fallback on fresh pods — so its committed
# reference rho embeds whatever bank state its pod held, while #2254 stages
# the canonical sha-pinned #779 banks (plan §4.4 bank-identity staging).
# Discriminator: evil passed ALL layers at 5e-3 while only the bank-loaded
# behaviors deviated (max rel 1.22e-2, mixed sign) => kernels/template/model
# identical; the residual is reference-side bank provenance, not a rig fault.
# 2e-2 keeps ~1.6x margin over the observed max while a real geometry error
# (wrong position / normalization / layer) still HALTs by >10x. Cells above
# 5e-3 but within the banked tolerance are recorded provenance_waived=true —
# never silently equal-treated.
RHO_2220_JSON_REL = "eval_results/issue_2220/norm_probe/rho_by_layer.json"
RHO_2220_CONE = "eval_results/issue_2220/norm_probe"
RHO_PARITY_RTOL = 5e-3
RHO_PARITY_RTOL_BANKED = 2e-2
RHO_PARITY_BANKED_BEHAVIORS = frozenset({"hallucination", "sycophancy"})

# #2220 read-direction bank (Result 0 cosines; plan §10 reuse row).
RW2220_DIR_PREFIX = "issue2220_readwrite/directions"
RW2220_READ_SLUGS = ("mapread_ctx", "mapread_prefix")
# Data-repo revision pin for the #2220 read-direction bank (plan §4.1 "at the
# manifest pin"): resolved main sha 2026-08-12, manifest `file_exists` verified
# at this revision. A #2220 re-upload can no longer silently change Result 0
# cosines (review minor g2 — every other cross-issue input is rev-pinned).
RW2220_DIR_REVISION = "a1935e1957526bf42762eb5ce4047e5539c04f1b"

# Timing pilot (gate-1 input; plan §7 gate 1 + §9 norm_probe row).
PILOT_LAYER = 14
PILOT_DOSE_C = 1.0
PILOT_N_QUESTIONS = 10  # localize Q1 grain (production cell shape)
PILOT_N_DRAWS = 3
PILOT_SEED = 42
GEN_MAX_NEW_TOKENS = 2048
# Plan §4.2 grid arithmetic (completions), for the pilot's extrapolation block:
# localize 1,155 cells x 30; decisive 75 x 200; patch 39 x 200; baseline 6 x 200.
PLAN_COMPLETIONS = {
    "localize": 1155 * 30,
    "decisive": 75 * 200,
    "patch": 39 * 200,
    "baseline_ceiling": 6 * 200,
}
GATE1_THRESHOLD_GPU_H = 60.0  # plan §7 gate 1 (decision is orchestrator-owned)

# Verbatim #1615 GCV grid (pinv_topk_contexts.py L85; referenced by
# ridge_fit_matrix below — part of the verbatim copy).
LAMBDAS = np.logspace(-2, 4, 13)


# ---------------------------------------------------------------------------
# VERBATIM #1615 ridge fit (provenance: eval_results/issue_779/
# pinv_topk_contexts/pinv_topk_contexts.py L92-120; do not edit)
# ---------------------------------------------------------------------------


def ridge_fit_matrix(X_train, Y_train):
    """VERBATIM from pinv_direction_read.py: replicate fit_h.ridge_fit_predict
    internals, returning W (d, D_out) + standardization params + GCV lambda +
    standardized-X singular values. Reproduces F.ridge_fit_predict to machine
    precision (verified there); the recon-R2 cross-check below re-confirms it."""
    Xtr = np.asarray(X_train, dtype=np.float64)
    Ytr = np.asarray(Y_train, dtype=np.float64)
    n = Xtr.shape[0]
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu
    U, s, Vt = np.linalg.svd(Xtr_n, full_matrices=False)
    s2 = s**2
    UtY = U.T @ Ytr_c
    best_lam, best_gcv = LAMBDAS[0], np.inf
    for lam in LAMBDAS:
        filt = s2 / (s2 + lam)
        Yhat_tr = U @ (filt[:, None] * UtY)
        rss = float(np.sum((Ytr_c - Yhat_tr) ** 2))
        dof = float(np.sum(filt))
        denom = (n - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else np.inf
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, lam
    filt = s / (s2 + best_lam)
    W = (Vt.T * filt) @ UtY  # (d, D_out)
    return {"W": W, "xmu": xmu, "xsd": xsd, "ymu": ymu, "s": s, "lam": float(best_lam)}


# ---------------------------------------------------------------------------
# Pure pre-image algebra (CPU-tested; tests/test_issue2254_driver.py)
# ---------------------------------------------------------------------------


def kstar_from_fit(s, lam) -> int:
    """#1615 pre-registered rule: k* = #{i: s_i^2 >= lambda}, s = the FIT's
    standardized-X singular values (pinv_direction_read.py L167)."""
    return int(np.sum(np.asarray(s, dtype=np.float64) ** 2 >= float(lam)))


def map_svd(W):
    """M = W.T (v = M c_std; pinv_direction_read.py L163-164) + its SVD."""
    M = np.asarray(W, dtype=np.float64).T
    Um, Sm, Vmt = np.linalg.svd(M, full_matrices=False)
    return M, Um, Sm, Vmt


def preimage_w(Um, Sm, Vmt, r_b, k: int):
    """Truncated pinv direction (pinv_direction_read.py L171-174):
    w = V_k diag(1/s_k) U_k^T r_B, standardized-context frame."""
    kk = int(min(k, Sm.shape[0]))
    if kk <= 0:
        raise ValueError(f"preimage_w: k={k} leaves no components (degenerate map)")
    coeff = (Um.T @ np.asarray(r_b, dtype=np.float64))[:kk] / Sm[:kk]
    return Vmt[:kk].T @ coeff


def destandardized_direction(xsd, w):
    """De-standardization fold (plan §4.1 row 1): d = normalize(xsd * w) —
    the raw residual-space edit whose predicted answer shift is P_k*(r_B)."""
    d = np.asarray(xsd, dtype=np.float64) * np.asarray(w, dtype=np.float64)
    nrm = float(np.linalg.norm(d))
    if not (np.isfinite(nrm) and nrm > 0.0):
        raise ValueError(f"destandardized_direction: degenerate norm {nrm!r}")
    return d / nrm


def topk_projection(Um, r_b, k: int):
    """P_k(r_B) = U_k U_k^T r_B — projection onto the rank-k column space of M."""
    kk = int(min(max(k, 0), Um.shape[1]))
    u = Um[:, :kk]
    return u @ (u.T @ np.asarray(r_b, dtype=np.float64))


def frame_fold_cos(M, Um, xsd, d_pre, r_b, k: int) -> float:
    """HALT-class frame-fold test (plan §4.1): cos(M @ (d_pre / xsd), P_k(r_B))."""
    lhs = M @ (np.asarray(d_pre, dtype=np.float64) / np.asarray(xsd, dtype=np.float64))
    rhs = topk_projection(Um, r_b, k)
    den = float(np.linalg.norm(lhs) * np.linalg.norm(rhs))
    if den <= 0.0:
        raise ValueError("frame_fold_cos: degenerate operands")
    return float(lhs @ rhs / den)


def proj_fraction(Um, r_b, k: int) -> float:
    """||P_k* r_B|| / ||r_B|| — how much of r_B the rank-k* map can reach."""
    r = np.asarray(r_b, dtype=np.float64)
    kk = int(min(max(k, 0), Um.shape[1]))
    return float(np.linalg.norm((Um.T @ r)[:kk]) / (np.linalg.norm(r) + 1e-300))


def shuffled_direction_bundle(fit_shuf: dict, kstar_real: int, r_b) -> dict:
    """Shuffled-map pre-image pair (plan §4.1 row 5 + the critic-mandated pin).

    PRIMARY = the registered exact construction (the shuffled fit's OWN GCV
    lambda + OWN k*); ALSO a matched-k* variant (truncate the shuffled SVD at
    the REAL map's k*) as a diagnostic direction only. k*_shuffled == 0 =>
    the STEERING shuffled direction falls back to the matched-k* variant,
    flagged loudly (never normalize(0)).
    """
    _Ms, Ums, Sms, Vmts = map_svd(fit_shuf["W"])
    k_shuf = kstar_from_fit(fit_shuf["s"], fit_shuf["lam"])
    k_matched = int(min(max(kstar_real, 1), Sms.shape[0]))
    matched = destandardized_direction(fit_shuf["xsd"], preimage_w(Ums, Sms, Vmts, r_b, k_matched))
    if k_shuf == 0:
        return {
            "kstar_shuffled": 0,
            "d_preshuf_primary": None,
            "d_preshuf_matched": matched,
            "d_preshuf_steering": matched,
            "fallback_matched_kstar": True,
        }
    primary = destandardized_direction(fit_shuf["xsd"], preimage_w(Ums, Sms, Vmts, r_b, k_shuf))
    return {
        "kstar_shuffled": k_shuf,
        "d_preshuf_primary": primary,
        "d_preshuf_matched": matched,
        "d_preshuf_steering": primary,
        "fallback_matched_kstar": False,
    }


def diff_of_means_direction(pos_acts, neg_acts):
    """Direction 3 (plan §4.1 row 3): per-layer diff of means at the last
    context token, unit-normalized. Inputs (n_pos, L, H) / (n_neg, L, H)."""
    pos = np.asarray(pos_acts, dtype=np.float64)
    neg = np.asarray(neg_acts, dtype=np.float64)
    assert pos.ndim == 3 and neg.ndim == 3 and pos.shape[1:] == neg.shape[1:], (
        pos.shape,
        neg.shape,
    )
    diff = pos.mean(axis=0) - neg.mean(axis=0)  # (L, H)
    nrm = np.linalg.norm(diff, axis=1, keepdims=True)
    if not (np.isfinite(nrm).all() and (nrm > 0.0).all()):
        raise ValueError(f"diff_of_means_direction: degenerate layer norms {nrm.ravel()!r}")
    return diff / nrm


def random_direction(d, *, seed, n_avg=N_RANDOM_SEEDS):
    """Matched-norm random unit vector, mean over ``n_avg`` seeds.

    VERBATIM from issue2220_readwrite.py L345-361 (plan §4.1 row 4 "the #2220
    construction"): each seed draws a Gaussian, the mean over seeds is
    re-normalized to unit norm."""
    acc = np.zeros(d, dtype=np.float64)
    for s in range(n_avg):
        rng = np.random.default_rng(seed * 1000 + s)
        v = rng.standard_normal(d)
        acc += v / float(np.linalg.norm(v))
    nrm = float(np.linalg.norm(acc))
    return acc / nrm


def unit_rows(mat):
    """Row-normalize a (L, H) array, failing loud on a degenerate row."""
    m = np.asarray(mat, dtype=np.float64)
    nrm = np.linalg.norm(m, axis=1, keepdims=True)
    if not (np.isfinite(nrm).all() and (nrm > 0.0).all()):
        raise ValueError("unit_rows: degenerate row norm")
    return m / nrm


def r2_score_multi(pred, true) -> dict:
    """R2 + mean cosine, the fit_h.reconstruction_metrics convention (SS_tot
    around true.mean(0)); local copy so ProcessPool workers stay light."""
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mu = true.mean(0)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - mu) ** 2))
    r2 = float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot
    num = np.sum(pred * true, axis=1)
    den = (np.linalg.norm(pred, axis=1) + 1e-12) * (np.linalg.norm(true, axis=1) + 1e-12)
    cos = float(np.mean(num / den))
    return {"r2": r2, "mean_cosine": cos}


def assert_sha256(path: Path, expected: str, *, what: str = "") -> None:
    """Fail-loud sha256 content pin (plan §4.4 bank-identity staging)."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise RuntimeError(
            f"sha256 mismatch for {what or path}: expected {expected[:16]}..., "
            f"got {digest[:16]}... — the staged e1 asset does not match the plan pin; "
            "refusing to proceed (the load_e1_assets regen fallback must stay unreachable)"
        )


def assert_e1_eval_bank(assets: dict, behavior: str) -> None:
    """`eval_questions` present + length 20 per trait (plan §4.4: the loader
    itself asserts only instruction/extraction_questions/eval_prompt —
    generation.py L441-443 — so the driver asserts the eval bank)."""
    qs = assets.get("eval_questions")
    if not isinstance(qs, list) or len(qs) != N_EVAL_QUESTIONS:
        n = len(qs) if isinstance(qs, list) else None
        raise RuntimeError(
            f"[{behavior}] e1 assets eval_questions invalid: expected list of "
            f"{N_EVAL_QUESTIONS}, got {type(qs).__name__} len={n} — bank identity "
            "with #779/#2220 is broken (plan §4.4)"
        )


_E1_STAGED_OK: set[str] = set()


def _assert_e1_staged(behavior: str) -> None:
    """Fail-loud e1 bank-identity gate at EVERY `load_e1_assets` consumer.

    The upstream loader carries a Sonnet-REGENERATION fallback on a missing/
    malformed artifact (issue_1739/generation.py L416-441) that would silently
    swap in different questions; `phase_stage_inputs` makes it unreachable in
    ITS process only. This helper re-asserts presence + the plan §4.4 sha256
    pin BEFORE any consumer-phase load — inherited by every call site through
    the `_eval_questions` / `_extraction_contexts` / `_positive_instructions`
    wrappers, so `capture_directions`, `norm_probe`, and every unit-3 phase
    (incl. the fresh pod B, where stage_inputs is not in the §9 phase table)
    cannot reach the regen fallback (review blocker g2, round 1). Evil is
    code-resident (paper-verbatim EVIL_ARTIFACTS — no file to pin). Cached
    per process: one hash per behavior.
    """
    expected = E1_ASSET_SHA256.get(behavior)
    if expected is None or behavior in _E1_STAGED_OK:
        return
    _ensure_repo_root_on_syspath()
    from scripts.issue779_common import _artifacts_dir

    target = _artifacts_dir() / f"{behavior}.json"
    if not target.is_file():
        raise RuntimeError(
            f"e1 asset {target} MISSING — run --phase stage_inputs first; refusing "
            "to let load_e1_assets reach its Sonnet-regeneration fallback (plan §4.4 "
            "bank identity with #779/#2220)"
        )
    assert_sha256(target, expected, what=f"e1 asset {behavior}.json")
    _E1_STAGED_OK.add(behavior)


# ---------------------------------------------------------------------------
# shared: paths, sentinel, breadcrumbs (issue2220_readwrite conventions)
# ---------------------------------------------------------------------------


def _sha8(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:8]


def _breadcrumb(phase: str, **kw) -> None:
    kv = " ".join(f"{k}={v}" for k, v in kw.items())
    print(f"[phase={phase}] {kv}", flush=True)


def _progress(phase: str, k: int, n: int, key: str, t0: float) -> None:
    print(f"[{phase}] unit {k}/{n} {key} elapsed={time.time() - t0:.1f}s", flush=True)


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    os.replace(tmp, path)


def _run_metadata(extra: dict | None = None) -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    md = {
        "experiment": "issue2254_preimage",
        "base_model": MODEL_NAME,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pass_b_revision": HF_REV,
    }
    md.update(as_metadata_dict(git_provenance()))
    if extra:
        md.update(extra)
    return md


def _write_sentinel(out_root: Path, phase: str, status: str, extra: dict | None = None) -> Path:
    """Per-phase sentinel file (/workspace/logs/issue-2254-<phase>.json),
    observed by the ORCHESTRATOR via file presence / direct reads — the
    envelope deliberately mirrors the #2220 parent convention and carries
    NONE of poll_pipeline._SENTINEL_REQUIRED_KEYS, so `_parse_sentinel`
    SKIPS it (persisted concern `sentinel-envelope-poller-drain`; the
    dispatch gate must not rely on the poller's envelope drain). Pod-side
    code NEVER shells to task.py."""
    logs = Path(os.environ.get("EPM_SENTINEL_DIR", "/workspace/logs"))
    payload = {"issue": ISSUE, "phase": phase, "status": status, "out_root": str(out_root)}
    if extra:
        payload.update(extra)
    try:
        logs.mkdir(parents=True, exist_ok=True)
        p = logs / f"issue-{ISSUE}-{phase}.json"
        _write_json_atomic(p, payload)
        return p
    except OSError as exc:  # sentinel dir absent off-pod (VM smoke) -> log, never crash
        logger.info("[sentinel] %s not writable (%s); skipping", logs, type(exc).__name__)
        return Path("/dev/null")


def _out_root(args) -> Path:
    return Path(args.out_root)


def _assert_phase_headroom(out_root: Path, need_gb: float, phase: str) -> None:
    """Per-write-phase disk headroom (plan §9 disk row; resume-aware sizing is
    N/A here — every unit-2 phase's writes are far below the floor passed)."""
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    assert_out_root_headroom(out_root, need_gb, phase=phase)


def _ensure_git_input(rel_file: str, cone: str) -> Path:
    """Materialize a committed cross-issue git input on partial-clone pods.

    ``committed at HEAD`` != ``present``: partial-clone pods' default sparse
    cones exclude eval_results/ (gotchas.md "Partial-clone pods", #2211), and
    sparse worktrees exclude other issues' eval_results. Idempotent skip when
    present; else `git sparse-checkout add <cone>` in the repo root this
    driver resolves, then fail-loud verify (issue2220_readwrite
    `_ensure_parent_issue_cones` pattern).
    """
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "pyproject.toml").exists(), f"repo-root sentinel missing at {repo_root}"
    target = repo_root / rel_file
    if target.is_file():
        return target
    proc = subprocess.run(
        ["git", "sparse-checkout", "add", cone],
        cwd=repo_root,
        env={**os.environ},
        capture_output=True,
        text=True,
    )
    logger.info(
        "[cone-ensure] git sparse-checkout add %s -> rc=%d%s",
        cone,
        proc.returncode,
        f" stderr={proc.stderr.strip()!r}" if proc.returncode != 0 else "",
    )
    if not target.is_file():
        raise FileNotFoundError(
            f"Committed git input missing after cone-ensure: {rel_file}. Run "
            f"`git -C {repo_root} sparse-checkout add {cone}` (partial-clone pods + "
            "sparse worktrees exclude eval_results/ from the default cones) or `git pull`."
            + (f" git stderr: {proc.stderr.strip()!r}" if proc.stderr.strip() else "")
        )
    return target


# ---------------------------------------------------------------------------
# pass-B bundle (plan §10 reuse row; realized-keys assert = check (c))
# ---------------------------------------------------------------------------

_BUNDLE_CACHE: dict | None = None


def _load_pass_b_bundle() -> dict:
    """Download + validate the #779 pass-B train bundle at the pinned revision.

    Realized-keys assert (artifact-reuse check (c), plan §10) against the
    OBSERVED schema at rev ``037fcbb2`` — {cx_last, cx_mean, v_x, layers,
    metadata, source}, tensors (N, 28, 3584) float32, zero NaN/Inf, layers ==
    list(range(28)). The uploaded bundle carries NO ``prompts`` key BY
    CONSTRUCTION: the producer's content-hygiene sanitizer strips raw LMSYS
    text before any analysis_tensors/ upload (issue779_collect.py
    ``_sanitize_for_analysis_tensors`` + ``_assert_no_raw_text_under``); the
    plan §10 "prompts" claim was schema-from-producer-code, not from the
    artifact (#2061 shape). Prompt text is reconstructed separately via the
    pinned #823/#952/#1615 LMSYS replay (see ``phase_stage_inputs``).
    Returns {"cx_last": tensor, "v_x": tensor, "source": str, "n_rows": N}.
    Memoized per process (stage_inputs A8 leg + fit_maps share the load).
    """
    global _BUNDLE_CACHE
    if _BUNDLE_CACHE is not None:
        return _BUNDLE_CACHE
    import torch
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    path = hub.retry_transient(
        lambda: hf_hub_download(
            repo_id=HF_DATA_REPO,
            filename=PASS_B_FILE,
            repo_type="dataset",
            revision=HF_REV,
        ),
        what="fetch #779 pass-B bundle",
    )
    # weights_only=False: sha-pinned self-produced bundle carrying non-tensor
    # metadata alongside tensors — the torch>=2.6 weights_only default cannot
    # load it, and the revision pin is the trust boundary.
    tb = torch.load(path, map_location="cpu", weights_only=False)
    missing = {"cx_last", "v_x", "layers", "source"} - set(tb.keys())
    if missing:
        raise RuntimeError(
            f"pass-B bundle at rev {HF_REV[:12]} missing keys {sorted(missing)} "
            f"(realized keys: {sorted(tb.keys())})"
        )
    layers = list(tb["layers"])
    if layers != list(range(N_LAYERS)):
        raise RuntimeError(f"pass-B bundle layers != range(28): {layers[:5]}...")
    cx, vx = tb["cx_last"], tb["v_x"]
    n = int(cx.shape[0])
    for name, t in (("cx_last", cx), ("v_x", vx)):
        if tuple(t.shape) != (n, N_LAYERS, HIDDEN_DIM):
            raise RuntimeError(f"pass-B {name} shape {tuple(t.shape)} != ({n}, 28, {HIDDEN_DIM})")
        if not torch.isfinite(t).all():
            raise RuntimeError(f"pass-B {name} carries NaN/Inf")
    logger.info("[pass-b] realized N=%d rows, 28 layers, H=%d (rev %s)", n, HIDDEN_DIM, HF_REV[:12])
    _BUNDLE_CACHE = {"cx_last": cx, "v_x": vx, "source": str(tb["source"]), "n_rows": n}
    return _BUNDLE_CACHE


def _load_rb_all() -> dict[str, np.ndarray]:
    """#779 r_B bank at the pin -> {behavior: (28, H) float64} (NOT normalized)."""
    from explore_persona_space.experiments.issue_1739 import store_io

    rb_bank, trait_names = store_io.load_rb_bank(revision=HF_REV)
    out: dict[str, np.ndarray] = {}
    for b in BEHAVIORS:
        if b not in trait_names:
            raise RuntimeError(f"behavior {b} absent from r_B bank traits {trait_names}")
        ti = trait_names.index(b)
        out[b] = np.asarray(rb_bank[:, ti, :], dtype=np.float64)  # (28, H)
    return out


# ---------------------------------------------------------------------------
# phase: stage_inputs (CPU pod-side; sha-pinned e1 staging + A8 gate)
# ---------------------------------------------------------------------------


def _stage_e1_assets() -> dict:
    """Stage the sha-pinned e1 asset JSONs into data/issue_779/artifacts/.

    Downloads from `issue2254_preimage/inputs/e1_assets/` (uploaded at
    dispatch), asserts sha256 == the plan pins BEFORE any load_e1_assets call
    — the Sonnet regeneration fallback inside load_e1_assets is thereby
    unreachable (plan §4.4). Idempotent: a present local file with the pinned
    sha is kept; wrong-sha local copies are re-downloaded then re-asserted.
    """
    from explore_persona_space.orchestrate import hub

    # Write EXACTLY where the loader reads: issue779_common._artifacts_dir()
    # anchors on its OWN file (checkout root), never the cwd.
    _ensure_repo_root_on_syspath()
    from scripts.issue779_common import _artifacts_dir

    e1_dir = _artifacts_dir()
    e1_dir.mkdir(parents=True, exist_ok=True)
    record = {}
    for behavior, expected in E1_ASSET_SHA256.items():
        target = e1_dir / f"{behavior}.json"
        if target.is_file():
            got = hashlib.sha256(target.read_bytes()).hexdigest()
            if got == expected:
                logger.info("[stage_inputs] %s already staged (sha ok)", target)
                record[behavior] = {"path": str(target), "sha256": got, "downloaded": False}
                continue
            logger.warning("[stage_inputs] %s present with WRONG sha; re-staging", target)
            target.unlink()
        # Canonical (non-smoke) input prefix: the dispatch step uploads there
        # once; smoke runs consume the same pinned inputs.
        hub.stage_hub_file(
            HF_DATA_REPO,
            f"{E1_INPUT_PREFIX}/{behavior}.json",
            target,
            repo_type="dataset",
        )
        assert_sha256(target, expected, what=f"e1 asset {behavior}.json")
        record[behavior] = {"path": str(target), "sha256": expected, "downloaded": True}
    return record


def phase_stage_inputs(args) -> None:
    """Stage sha-pinned e1 assets + run the A8 disjointness gate (pre-spend).

    A8 (plan §4.4/§6): every behavior's eval bank must be disjoint from (a)
    the pooled #779 extraction sets and (b) the pass-B LMSYS map-fit prompts
    (the corpus leg of the reused issue2220 gate, with our corpus threaded
    through its ``corpus_texts=`` parameter).
    """
    out_root = _out_root(args)
    _assert_phase_headroom(out_root, 1.0, "stage_inputs")
    behaviors = list(args.behaviors)
    _breadcrumb("stage_inputs", behaviors=len(behaviors))

    staged = _stage_e1_assets()

    # eval_questions present + length 20 per trait — BEFORE the A8 gate reads
    # them (the loader's own asserts do not cover the eval bank).
    _ensure_repo_root_on_syspath()
    from explore_persona_space.experiments.issue_1739.generation import load_e1_assets

    for b in behaviors:
        assets = load_e1_assets(b)
        assert_e1_eval_bank(assets, b)
        assert len(assets["instruction"]) >= N_INSTRUCTION_PAIRS, b
        assert len(assets["extraction_questions"]) >= N_EXTRACTION_QUESTIONS, b

    # A8 gate, reused from issue2220_readwrite (import — the helpers it uses
    # are behavior-generic and its Q2=20 floor matches ours); corpus leg =
    # the pass-B LMSYS prompts (normalized, set-compared, never logged —
    # content hygiene). The uploaded bundle carries NO prompt text (producer
    # sanitizer strips it), so the prompts are RECONSTRUCTED via the pinned
    # #823/#952/#1615 replay: first N non-empty first-user-turns of
    # lmsys/lmsys-chat-1m @ LMSYS_REVISION — verified 1:1 with the bundle rows
    # by #1615 (n_train=5000, no kept_idx drops).
    import scripts.issue2220_readwrite as rw2220
    from scripts.issue952_stats import _reconstruct_lmsys_prompts

    bundle = _load_pass_b_bundle()
    if bundle["source"] != "lmsys/lmsys-chat-1m":
        raise RuntimeError(
            f"A8: pass-B bundle source {bundle['source']!r} != 'lmsys/lmsys-chat-1m' — "
            "the pinned LMSYS replay recipe does not apply to a fallback-source bundle"
        )
    prompts = _reconstruct_lmsys_prompts(bundle["n_rows"])
    if len(prompts) != bundle["n_rows"]:
        raise RuntimeError(f"A8: replay yielded {len(prompts)} prompts != N={bundle['n_rows']}")
    corpus_texts = {rw2220._norm_question(p) for p in prompts if p}
    if not corpus_texts:
        raise RuntimeError("A8: LMSYS replay yielded 0 prompt texts")
    record = rw2220._assert_eval_bank_disjoint(behaviors, corpus_texts=corpus_texts)
    record["corpus_source"] = (
        f"pass-B LMSYS prompts, #823/#952/#1615 pinned replay (bundle {PASS_B_FILE} @ "
        f"{HF_REV[:12]} carries no prompt text by producer-sanitizer construction)"
    )
    record["n_bundle_rows"] = bundle["n_rows"]
    record["staged_e1_assets"] = staged

    _write_json_atomic(out_root / "stage_inputs" / "disjointness.json", _run_metadata(record))
    _write_sentinel(out_root, "stage_inputs", "done", {"n_corpus_texts": len(corpus_texts)})
    _breadcrumb("stage_inputs", status="done", behaviors=len(behaviors))


# ---------------------------------------------------------------------------
# phase: fit_maps (CPU pod-side; 28 real + 28 shuffled float64 SVD fits)
# ---------------------------------------------------------------------------


def _fit_layer_worker(task: dict) -> dict:
    """One layer's real + shuffled fits + pre-image constructions (ProcessPool
    unit; all float64 heavy math stays in the worker, W returned fp32).

    Held-out mapping-baselines report (90/10) precedes the production refit on
    ALL rows (the #1615 grain) — plan §4.1 mapping-baselines duty.
    """
    from explore_persona_space.analysis.mapping_baselines import (
        identity_bias_predict,
        knn_retrieval,
    )

    t0 = time.time()
    layer = int(task["layer"])
    X = np.asarray(task["x16"], dtype=np.float64)
    Y = np.asarray(task["y16"], dtype=np.float64)
    tr_idx = np.asarray(task["tr_idx"])
    ev_idx = np.asarray(task["ev_idx"])
    perm = np.asarray(task["perm_shuffle"])
    rb_rows: dict[str, np.ndarray] = task["rb_rows"]  # {behavior: (H,) float64}

    # -- held-out mapping-baselines report (fit on 90%, read on 10%) --------
    fit_ho = ridge_fit_matrix(X[tr_idx], Y[tr_idx])
    x_ev_n = (X[ev_idx] - fit_ho["xmu"]) / fit_ho["xsd"]
    pred_map = x_ev_n @ fit_ho["W"] + fit_ho["ymu"]
    heldout = {
        "n_train": int(tr_idx.size),
        "n_eval": int(ev_idx.size),
        "map": r2_score_multi(pred_map, Y[ev_idx]),
        "identity_bias": r2_score_multi(
            identity_bias_predict(X[tr_idx], Y[tr_idx], X[ev_idx]), Y[ev_idx]
        ),
        "knn": {
            metric: knn_retrieval(pred_map, Y[ev_idx], ks=KNN_KS, metric=metric)
            for metric in ("euclidean", "cosine")
        },
        "knn_chance_at_10": 10.0 / float(ev_idx.size),
    }

    # -- production refit on ALL rows (the #1615 grain) ----------------------
    fit = ridge_fit_matrix(X, Y)
    xn = (X - fit["xmu"]) / fit["xsd"]
    recon = r2_score_multi(xn @ fit["W"] + fit["ymu"], Y)
    kstar = kstar_from_fit(fit["s"], fit["lam"])
    if kstar <= 0:
        raise RuntimeError(f"L{layer}: real map k*=0 — degenerate fit (lam={fit['lam']})")
    M, Um, Sm, Vmt = map_svd(fit["W"])

    # -- shuffled-map control fit (row-permuted Y, one shared permutation) --
    fit_shuf = ridge_fit_matrix(X, Y[perm])
    kstar_shuf = kstar_from_fit(fit_shuf["s"], fit_shuf["lam"])

    per_behavior: dict[str, dict] = {}
    for b, r_b in rb_rows.items():
        w = preimage_w(Um, Sm, Vmt, r_b, kstar)
        d_raw = np.asarray(fit["xsd"], dtype=np.float64) * w
        d_pre = destandardized_direction(fit["xsd"], w)
        ff = frame_fold_cos(M, Um, fit["xsd"], d_pre, r_b, kstar)
        shuf = shuffled_direction_bundle(fit_shuf, kstar, r_b)
        per_behavior[b] = {
            "d_pre": d_pre.astype(np.float32),
            "d_preshuf_steering": shuf["d_preshuf_steering"].astype(np.float32),
            "d_preshuf_matched": shuf["d_preshuf_matched"].astype(np.float32),
            "frame_fold_cos": float(ff),
            "proj_frac_rb": proj_fraction(Um, r_b, kstar),
            # NOT "d_pre_raw_norm": the fit-report composition strips direction
            # VECTORS by `k.startswith("d_")` — this scalar diagnostic must
            # survive into the persisted fit_report.json (review minor g2).
            "raw_norm_d_pre": float(np.linalg.norm(d_raw)),
            "kstar_shuf_fallback": bool(shuf["fallback_matched_kstar"]),
        }

    return {
        "layer": layer,
        "lam": float(fit["lam"]),
        "kstar": int(kstar),
        "recon": recon,
        "heldout": heldout,
        "lam_shuffled": float(fit_shuf["lam"]),
        "kstar_shuffled": int(kstar_shuf),
        "per_behavior": per_behavior,
        "W32": np.asarray(fit["W"], dtype=np.float32),
        "xmu": np.asarray(fit["xmu"], dtype=np.float64),
        "xsd": np.asarray(fit["xsd"], dtype=np.float64),
        "ymu": np.asarray(fit["ymu"], dtype=np.float64),
        "s": np.asarray(fit["s"], dtype=np.float64),
        "fit_wall_s": float(time.time() - t0),
    }


def _parity_gate(records: dict[int, dict], behaviors: list[str]) -> dict:
    """HALT-class structural parity vs the committed #1615 read (plan §4.1).

    At each behavior's frozen layer the refit must reproduce the committed
    pinv_direction_read.json lambda, k*, and recon R2. A mismatch means the
    bundle or recipe drifted and BLOCKS production.
    """
    committed = json.loads(_ensure_git_input(PARITY_JSON_REL, PARITY_CONE).read_text())
    out: dict[str, dict] = {}
    checked = 0
    for b in behaviors:
        frozen = FROZEN_LAYER[b]
        ref = committed["traits"][b]
        if (
            int(ref["read_out_layer"]) != frozen
            or int(ref["k_ridge_estimable_prereg"]) != (KSTAR_PREREG[b])
        ):
            raise RuntimeError(
                f"parity reference file mismatch for {b}: layer/k* "
                f"({ref['read_out_layer']}/{ref['k_ridge_estimable_prereg']}) != plan pins "
                f"({frozen}/{KSTAR_PREREG[b]}) — wrong committed file?"
            )
        if frozen not in records:
            logger.info("[fit_maps] parity: %s frozen layer L%d not in run layers", b, frozen)
            continue
        rec = records[frozen]
        row = {
            "layer": frozen,
            "lam_committed": float(ref["ridge_lambda"]),
            "lam_refit": rec["lam"],
            "kstar_committed": int(ref["k_ridge_estimable_prereg"]),
            "kstar_refit": rec["kstar"],
            "recon_r2_committed": float(ref["recon_ridge"]["r2"]),
            "recon_r2_refit": rec["recon"]["r2"],
        }
        problems = []
        if not np.isclose(row["lam_refit"], row["lam_committed"], rtol=1e-9, atol=0.0):
            problems.append(f"lambda {row['lam_refit']} != {row['lam_committed']}")
        if row["kstar_refit"] != row["kstar_committed"]:
            problems.append(f"k* {row['kstar_refit']} != {row['kstar_committed']}")
        if abs(row["recon_r2_refit"] - row["recon_r2_committed"]) > PARITY_R2_ATOL:
            problems.append(
                f"recon R2 |{row['recon_r2_refit']:.6f} - {row['recon_r2_committed']:.6f}| "
                f"> {PARITY_R2_ATOL}"
            )
        if problems:
            raise RuntimeError(
                f"HALT: frozen-layer parity FAILED for {b}@L{frozen}: "
                + "; ".join(problems)
                + " — bundle or recipe drifted vs the committed #1615 read (plan §4.1)"
            )
        row["pass"] = True
        out[b] = row
        checked += 1
        logger.info(
            "[fit_maps] parity PASS %s@L%d: lam=%.4f k*=%d r2=%.6f",
            b,
            frozen,
            row["lam_refit"],
            row["kstar_refit"],
            row["recon_r2_refit"],
        )
    if checked == 0:
        raise RuntimeError(
            "HALT: no frozen parity layer in the run's layer set — the parity gate "
            f"cannot be skipped (frozen layers: {FROZEN_LAYER})"
        )
    return out


def phase_fit_maps(args) -> None:
    """Per-layer map fits + pre-image direction constructions (plan §4.1)."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    import torch

    out_root = _out_root(args)
    _assert_phase_headroom(out_root, 4.0, "fit_maps")  # maps ~1.5 GB fp32 + reports
    layers = sorted(int(x) for x in args.layers)
    behaviors = list(args.behaviors)
    _breadcrumb("fit_maps", layers=len(layers), behaviors=len(behaviors))

    bundle = _load_pass_b_bundle()
    n = bundle["n_rows"]
    rb_all = _load_rb_all()

    rng_split = np.random.default_rng(HELDOUT_SPLIT_SEED)
    perm_split = rng_split.permutation(n)
    n_ev = max(1, int(round(HELDOUT_FRAC * n)))
    ev_idx, tr_idx = perm_split[:n_ev], perm_split[n_ev:]
    perm_shuffle = np.random.default_rng(SEED_SHUFFLE).permutation(n)

    cx, vx = bundle["cx_last"], bundle["v_x"]
    tasks = []
    for ly in layers:
        tasks.append(
            {
                "layer": ly,
                "x16": np.ascontiguousarray(cx[:, ly, :].numpy()),
                "y16": np.ascontiguousarray(vx[:, ly, :].numpy()),
                "tr_idx": tr_idx,
                "ev_idx": ev_idx,
                "perm_shuffle": perm_shuffle,
                "rb_rows": {b: rb_all[b][ly] for b in behaviors},
            }
        )

    perlayer_dir = out_root / "maps" / "perlayer"
    perlayer_dir.mkdir(parents=True, exist_ok=True)
    records: dict[int, dict] = {}
    t0 = time.time()
    workers = max(1, min(int(args.fit_workers), len(tasks)))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_fit_layer_worker, t): t["layer"] for t in tasks}
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()  # fail-fast: worker exception propagates
            ly = rec["layer"]
            records[ly] = rec
            # persist the layer's map the moment it completes (durability)
            np.savez(
                perlayer_dir / f"L{ly:02d}.npz",
                W=rec["W32"],
                xmu=rec["xmu"],
                xsd=rec["xsd"],
                ymu=rec["ymu"],
                s=rec["s"],
                lam=np.float64(rec["lam"]),
                kstar=np.int64(rec["kstar"]),
                n_rows=np.int64(n),
                pass_b_revision=np.bytes_(HF_REV.encode()),
            )
            done += 1
            _progress("fit_maps", done, len(tasks), f"L{ly}", t0)

    # HALT gates (plan §4.1): frozen-layer parity + frame-fold per (b, layer).
    parity = _parity_gate(records, behaviors)
    fold_failures = [
        (b, ly, records[ly]["per_behavior"][b]["frame_fold_cos"])
        for ly in layers
        for b in behaviors
        if records[ly]["per_behavior"][b]["frame_fold_cos"] <= FRAME_FOLD_MIN_COS
    ]
    if fold_failures:
        raise RuntimeError(
            f"HALT: frame-fold test FAILED (cos <= {FRAME_FOLD_MIN_COS}) at "
            f"{[(b, ly, round(c, 6)) for b, ly, c in fold_failures[:6]]} — the "
            "de-standardization fold is broken (plan §4.1)"
        )

    # k*_shuffled == 0 steering fallback — recorded loudly (fit report +
    # sentinel extra + log); the ORCHESTRATOR posts the epm:progress note
    # (pod-side code never shells to task.py).
    fallback_layers = sorted(ly for ly in layers if records[ly]["kstar_shuffled"] == 0)
    if fallback_layers:
        logger.warning(
            "[fit_maps] k*_shuffled == 0 at layers %s — the STEERING shuffled "
            "direction falls back to the matched-k* variant there (plan §4.1 pin); "
            "orchestrator: post an epm:progress note naming these layers",
            fallback_layers,
        )

    # direction tensors for capture_directions (bank assembly input)
    dir_payload = {
        b: {
            "d_pre": torch.tensor(
                np.stack([records[ly]["per_behavior"][b]["d_pre"] for ly in layers])
            ),
            "d_preshuf": torch.tensor(
                np.stack([records[ly]["per_behavior"][b]["d_preshuf_steering"] for ly in layers])
            ),
            "d_preshuf_matchedk": torch.tensor(
                np.stack([records[ly]["per_behavior"][b]["d_preshuf_matched"] for ly in layers])
            ),
        }
        for b in behaviors
    }
    torch.save(
        {
            "layers": layers,
            "behaviors": behaviors,
            "directions": dir_payload,
            "kstar_shuffled_zero_layers": fallback_layers,
            "pass_b_revision": HF_REV,
        },
        out_root / "maps" / "preimage_directions.pt",
    )

    report = _run_metadata(
        {
            "n_rows_realized": n,
            "layers": layers,
            "behaviors": behaviors,
            "heldout_split": {
                "frac": HELDOUT_FRAC,
                "seed": HELDOUT_SPLIT_SEED,
                "n_train": int(tr_idx.size),
                "n_eval": int(ev_idx.size),
                "note": "pointwise 90/10 (explicit-iid LMSYS rows, plan §6); "
                "production maps refit on ALL rows (#1615 grain)",
            },
            "shuffle_seed": SEED_SHUFFLE,
            "parity_gate": parity,
            "frame_fold_min_cos": min(
                records[ly]["per_behavior"][b]["frame_fold_cos"] for ly in layers for b in behaviors
            ),
            "kstar_shuffled_zero_fallback_layers": fallback_layers,
            "per_layer": [
                {k: v for k, v in records[ly].items() if k not in ("W32", "xmu", "xsd", "ymu", "s")}
                | {
                    "per_behavior": {
                        b: {
                            k: v
                            for k, v in records[ly]["per_behavior"][b].items()
                            if not k.startswith("d_")
                        }
                        for b in behaviors
                    }
                }
                for ly in layers
            ],
        }
    )
    _write_json_atomic(out_root / "maps" / "fit_report.json", report)

    _upload_folder_to_hf(out_root / "maps", f"{_hf_prefix()}/analysis_tensors/maps_perlayer")
    _write_sentinel(
        out_root,
        "fit_maps",
        "done",
        {
            "n_layers": len(layers),
            "n_rows": n,
            "kstar_shuffled_zero_layers": fallback_layers,
        },
    )
    _breadcrumb("fit_maps", status="done", layers=len(layers), wall_s=round(time.time() - t0, 1))


def _upload_folder_to_hf(
    local_dir: Path, path_in_repo: str, allow: list[str] | None = None
) -> None:
    """One bulk upload_folder commit (never a per-file loop); fail-loud, retried
    (issue2220_readwrite `_upload_directions` pattern).

    `allow` is the upload_folder allow_patterns list (default covers the
    tensor/JSON artifact classes; judge packs pass ["*.jsonl", "*.json"] —
    plan-glob parity, #825)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    if allow is None:
        allow = ["*.pt", "*.json", "*.npz"]
    suffixes = tuple(p.lstrip("*") for p in allow)
    files = [p for p in local_dir.rglob("*") if p.is_file() and p.name.endswith(suffixes)]
    if not files:
        raise RuntimeError(f"[upload] no files under {local_dir} — refusing an empty upload")
    hub.assert_hub_dir_filecounts(str(local_dir), path_in_repo, allow_patterns=allow)
    api = HfApi()
    hub.retry_transient(
        lambda: api.upload_folder(
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            allow_patterns=allow,
        ),
        what=f"upload {path_in_repo}",
    )
    logger.info("[upload] %d files -> %s/%s", len(files), HF_DATA_REPO, path_in_repo)


# ---------------------------------------------------------------------------
# model loading (GPU phases)
# ---------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None


def _require_cuda(phase: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"--phase {phase} is a pod-side GPU phase (plan §9); CUDA is unavailable "
            "here — run it on the provisioned pod, never the shared VM"
        )


def _load_model_and_tokenizer():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("[model] loading %s (bf16)", MODEL_NAME)
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"  # generate_batch requires left-padding
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    _MODEL, _TOKENIZER = model, tok
    return model, tok


def _eval_questions(behavior: str) -> list[str]:
    """Disjoint 20-question eval bank via the reused issue2220 loader (fail-loud
    on a missing/short eval_questions key; question text never logged)."""
    _assert_e1_staged(behavior)  # regen fallback unreachable (blocker g2)
    _ensure_repo_root_on_syspath()
    import scripts.issue2220_readwrite as rw2220

    return rw2220._eval_questions(behavior)


def _contexts_for_questions(questions: list[str]) -> list[dict]:
    """steering context shape: {"system": None, "user": q}."""
    return [{"system": None, "user": q} for q in questions]


# ---------------------------------------------------------------------------
# phase: capture_directions (GPU; direction 3 + bank + Result 0 cosines)
# ---------------------------------------------------------------------------


def _extraction_contexts(behavior: str) -> tuple[list[dict], list[dict]]:
    """(pos, neg) chat-templated extraction contexts: 5 instruction pairs
    (system role) x 20 extraction questions (plan §4.1 row 3)."""
    _assert_e1_staged(behavior)  # regen fallback unreachable (blocker g2)
    _ensure_repo_root_on_syspath()
    from explore_persona_space.experiments.issue_1739.generation import load_e1_assets

    assets = load_e1_assets(behavior)
    assert_e1_eval_bank(assets, behavior)
    pairs = assets["instruction"][:N_INSTRUCTION_PAIRS]
    qs = assets["extraction_questions"][:N_EXTRACTION_QUESTIONS]
    pos = [{"system": p["pos"], "user": q} for p in pairs for q in qs]
    neg = [{"system": p["neg"], "user": q} for p in pairs for q in qs]
    return pos, neg


def _save_direction(dir_out: Path, behavior: str, slug: str, layer: int, vec, manifest: list):
    """Per-direction .pt + manifest entry (issue2220_readwrite convention)."""
    import torch

    v = torch.as_tensor(np.asarray(vec, dtype=np.float32))
    path = dir_out / f"{behavior}_{slug}_L{layer}.pt"
    torch.save({"direction": v, "behavior": behavior, "slug": slug, "layer": layer}, path)
    manifest.append(
        {
            "behavior": behavior,
            "slug": slug,
            "layer": layer,
            "path": path.name,
            "norm": float(np.linalg.norm(np.asarray(vec, dtype=np.float64))),
            "sha8": _sha8(np.asarray(vec, dtype=np.float32).round(6).tolist()),
        }
    )


def _cos(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-300))


def _load_2220_read_directions(behaviors: list[str]) -> dict:
    """#2220 read-direction bank (Result 0 reuse row): manifest + per-layer
    direction tensors from `issue2220_readwrite/directions/` on the data repo."""
    import torch

    from explore_persona_space.orchestrate import hub

    stage = Path("data/issue_2254/hf_dl/rw2220_directions")
    manifest_path = hub.stage_hub_file(
        HF_DATA_REPO,
        f"{RW2220_DIR_PREFIX}/manifest.json",
        stage / "manifest.json",
        repo_type="dataset",
        revision=RW2220_DIR_REVISION,
    )
    manifest = json.loads(Path(manifest_path).read_text())
    entries = [
        e
        for e in manifest["directions"]
        if e["behavior"] in behaviors and e["slug"] in RW2220_READ_SLUGS
    ]
    if not entries:
        raise RuntimeError(
            f"no #2220 read directions ({RW2220_READ_SLUGS}) found in the manifest "
            f"for behaviors {behaviors}"
        )
    out: dict[tuple[str, str, int], np.ndarray] = {}
    for e in entries:
        local = hub.stage_hub_file(
            HF_DATA_REPO,
            f"{RW2220_DIR_PREFIX}/{e['path']}",
            stage / e["path"],
            repo_type="dataset",
            revision=RW2220_DIR_REVISION,
        )
        # weights_only=False on a self-produced #2220 tensor, now behind the
        # revision pin (same convention as the pass-B bundle loader).
        payload = torch.load(local, map_location="cpu", weights_only=False)
        out[(e["behavior"], e["slug"], int(e["layer"]))] = np.asarray(
            payload["direction"], dtype=np.float64
        )
    return out


def phase_capture_directions(args) -> None:
    """Direction 3 capture + full direction bank + Result 0 cosines (plan §4.1)."""
    _require_cuda("capture_directions")
    from explore_persona_space.experiments.issue1415 import steering

    out_root = _out_root(args)
    _assert_phase_headroom(out_root, 2.0, "capture_directions")
    layers = sorted(int(x) for x in args.layers)
    behaviors = list(args.behaviors)
    _breadcrumb("capture_directions", behaviors=len(behaviors), layers=len(layers))

    pre_path = out_root / "maps" / "preimage_directions.pt"
    if not pre_path.is_file():
        raise FileNotFoundError(f"{pre_path} missing — run --phase fit_maps first")
    import torch

    pre = torch.load(pre_path, map_location="cpu", weights_only=False)
    if pre["layers"] != layers or [b for b in behaviors if b not in pre["behaviors"]]:
        raise RuntimeError(
            f"preimage_directions.pt grid mismatch: has layers={pre['layers'][:4]}.../"
            f"behaviors={pre['behaviors']}, requested {layers[:4]}.../{behaviors}"
        )

    model, tok = _load_model_and_tokenizer()
    rb_all = _load_rb_all()

    dir_out = out_root / "directions"
    dir_out.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[dict] = []
    geometry: dict[str, dict] = {}
    t0 = time.time()

    for bi, behavior in enumerate(behaviors, 1):
        pos_ctx, neg_ctx = _extraction_contexts(behavior)
        cap = steering.capture_vectors(model, tok, pos_ctx + neg_ctx, layers, batch_size=8)
        v_all = np.stack(
            [rec["v_c_context"].numpy() for rec in cap["per_context"]]
        )  # (2*n, L, H) fp32
        n_pos = len(pos_ctx)
        d_ctxext = diff_of_means_direction(v_all[:n_pos], v_all[n_pos:])  # (L, H) unit

        rb_unit = unit_rows(rb_all[behavior][layers])  # (L, H)
        d_pre = np.asarray(pre["directions"][behavior]["d_pre"], dtype=np.float64)
        d_preshuf = np.asarray(pre["directions"][behavior]["d_preshuf"], dtype=np.float64)
        d_matched = np.asarray(pre["directions"][behavior]["d_preshuf_matchedk"], dtype=np.float64)

        for li, ly in enumerate(layers):
            _save_direction(dir_out, behavior, "pre", ly, d_pre[li], manifest_entries)
            _save_direction(dir_out, behavior, "rb", ly, rb_unit[li], manifest_entries)
            _save_direction(dir_out, behavior, "ctxext", ly, d_ctxext[li], manifest_entries)
            _save_direction(dir_out, behavior, "preshuf", ly, d_preshuf[li], manifest_entries)
            _save_direction(
                dir_out, behavior, "preshuf_matchedk", ly, d_matched[li], manifest_entries
            )
            rnd = random_direction(HIDDEN_DIM, seed=SEED_RANDOM_BASE + ly)
            _save_direction(dir_out, behavior, "random", ly, rnd, manifest_entries)

        geometry[behavior] = {
            "layers": layers,
            "cos_pre_ctxext": [_cos(d_pre[i], d_ctxext[i]) for i in range(len(layers))],
            "cos_pre_rb": [_cos(d_pre[i], rb_unit[i]) for i in range(len(layers))],
            "cos_ctxext_rb": [_cos(d_ctxext[i], rb_unit[i]) for i in range(len(layers))],
            "cos_pre_preshuf": [_cos(d_pre[i], d_preshuf[i]) for i in range(len(layers))],
        }
        _progress("capture_directions", bi, len(behaviors), behavior, t0)

    # Result 0: cosines to the #2220 read-direction bank at its 5 layers.
    rw_dirs = _load_2220_read_directions(behaviors)
    vs_2220: dict[str, list[dict]] = {b: [] for b in behaviors}
    fam_by_behavior = {
        b: {
            "pre": np.asarray(pre["directions"][b]["d_pre"], dtype=np.float64),
            "rb": unit_rows(rb_all[b][layers]),
        }
        for b in behaviors
    }
    for (b, slug, ly), vec in sorted(rw_dirs.items()):
        if b not in vs_2220 or ly not in layers:
            continue
        li = layers.index(ly)
        row = {"rw2220_slug": slug, "layer": ly}
        for fam, arr in fam_by_behavior[b].items():
            row[f"cos_{fam}"] = _cos(arr[li], vec)
        # ctxext cosine needs the captured direction; recover from saved bank
        ctx_path = dir_out / f"{b}_ctxext_L{ly}.pt"
        ctx_vec = torch.load(ctx_path, map_location="cpu", weights_only=False)["direction"]
        row["cos_ctxext"] = _cos(np.asarray(ctx_vec, dtype=np.float64), vec)
        vs_2220[b].append(row)

    manifest = _run_metadata({"directions": manifest_entries, "layers": layers})
    _write_json_atomic(dir_out / "manifest.json", manifest)
    geo_payload = _run_metadata(
        {
            "geometry": geometry,
            "vs_2220_read_directions": vs_2220,
            "note": "all directions unit-normalized; preshuf = steering (fallback-"
            "resolved) shuffled pre-image; preshuf_matchedk = diagnostic only",
        }
    )
    _write_json_atomic(out_root / "directions" / "geometry_cosines.json", geo_payload)

    _upload_folder_to_hf(dir_out, f"{_hf_prefix()}/directions")
    _write_sentinel(out_root, "capture_directions", "done", {"n_dirs": len(manifest_entries)})
    _breadcrumb("capture_directions", status="done", n_dirs=len(manifest_entries))


# ---------------------------------------------------------------------------
# phase: norm_probe (GPU; rho_l at 28 layers + #2220 parity + timing pilot)
# ---------------------------------------------------------------------------


def _parity_rtol_for(behavior: str) -> tuple[float, bool]:
    """(rtol, banked) for the rho-parity gates: bank-loaded behaviors get
    RHO_PARITY_RTOL_BANKED (reference-side bank provenance — see the constants
    block rationale); code-resident behaviors keep the binding RHO_PARITY_RTOL."""
    banked = behavior in RHO_PARITY_BANKED_BEHAVIORS
    return (RHO_PARITY_RTOL_BANKED if banked else RHO_PARITY_RTOL), banked


def _rho_parity_assert(result: dict[str, dict[str, float]]) -> dict:
    """Rig-parity probe vs #2220's committed rho values at shared layers
    (plan §4.2). Matched single-row forward geometry => tight relative
    tolerance; a real failure (wrong position/layer/bank) is a >10% effect.
    Per-behavior tolerance via ``_parity_rtol_for``: evil (code-resident on
    both sides) is the binding 5e-3 kernel/recipe canary; bank-loaded
    behaviors assert at RHO_PARITY_RTOL_BANKED, with every cell above 5e-3
    but within the banked tolerance recorded ``provenance_waived: true``."""
    ref = json.loads(_ensure_git_input(RHO_2220_JSON_REL, RHO_2220_CONE).read_text())
    ref_rho = ref["rho_median_last_context_token"]
    checked, problems, n_waived = [], [], 0
    for behavior, ref_layers in ref_rho.items():
        ours = result.get(behavior)
        if ours is None:
            continue
        rtol, banked = _parity_rtol_for(behavior)
        for lkey, ref_val in ref_layers.items():
            if lkey not in ours:
                continue
            rel = abs(ours[lkey] - float(ref_val)) / max(abs(float(ref_val)), 1e-12)
            waived = banked and RHO_PARITY_RTOL < rel <= rtol
            n_waived += int(waived)
            checked.append(
                {
                    "behavior": behavior,
                    "layer": lkey,
                    "ours": ours[lkey],
                    "rw2220": ref_val,
                    "rel_dev": rel,
                    "tolerance_applied": rtol,
                    "banked": banked,
                    "provenance_waived": waived,
                }
            )
            if rel > rtol:
                problems.append(
                    f"{behavior}@{lkey}: {ours[lkey]:.4f} vs {ref_val:.4f} rel={rel:.2e} "
                    f"> rtol={rtol:.0e} (banked={banked})"
                )
    if not checked:
        raise RuntimeError(
            "HALT: rho parity probe found NO shared (behavior, layer) cells vs "
            f"{RHO_2220_JSON_REL} — rig parity cannot be skipped (plan §4.2)"
        )
    if problems:
        raise RuntimeError(
            f"HALT: rho parity vs #2220 FAILED (rtol {RHO_PARITY_RTOL} code-resident / "
            f"{RHO_PARITY_RTOL_BANKED} bank-loaded): " + "; ".join(problems)
        )
    logger.info(
        "[norm_probe] rho parity PASS: %d shared cells (rtol %.1e code-resident / %.1e "
        "bank-loaded; %d provenance-waived above %.1e)",
        len(checked),
        RHO_PARITY_RTOL,
        RHO_PARITY_RTOL_BANKED,
        n_waived,
        RHO_PARITY_RTOL,
    )
    return {
        "n_checked": len(checked),
        "rtol": RHO_PARITY_RTOL,
        "rtol_banked": RHO_PARITY_RTOL_BANKED,
        "banked_behaviors": sorted(RHO_PARITY_BANKED_BEHAVIORS),
        "n_provenance_waived": n_waived,
        "cells": checked,
    }


def _timing_pilot(args, model, tok, rho_result: dict) -> dict:
    """Gate-1 input: ONE production-shape steering cell, measured wall
    (plan §7 gate 1 basis; §9 norm_probe row). Direction = unit r_B @ L14 at
    the context vector, c=+1 (timing is direction-independent; r_B avoids a
    cross-phase dependency on the bank), 10 questions x 3 draws (localize
    grain), cap 2048, temperature 1.0, seed 42 — the verbatim #2220 protocol.
    """
    import torch

    from explore_persona_space.experiments.issue1415 import steering

    behavior = list(args.behaviors)[0]
    questions = _eval_questions(behavior)[: args.pilot_questions]
    contexts = _contexts_for_questions(questions)
    rb_all = _load_rb_all()
    rb_vec = rb_all[behavior][PILOT_LAYER]
    rb_unit = rb_vec / float(np.linalg.norm(rb_vec))
    rho = rho_result[behavior][f"L{PILOT_LAYER}"]
    alpha = PILOT_DOSE_C * rho

    delta = torch.tensor(rb_unit, dtype=torch.bfloat16, device=model.device)
    t0 = time.time()
    with steering.DeltaHook(
        model, layer=PILOT_LAYER, delta=delta, alpha=float(alpha), all_positions=False
    ) as hook:
        results = steering.generate_batch(
            model,
            tok,
            contexts,
            n=args.pilot_draws,
            hook=hook,
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            temperature=1.0,
            seed_base=PILOT_SEED,
        )
    wall = time.time() - t0
    n_completions = len(contexts) * args.pilot_draws
    s_per = wall / n_completions
    # cap-hit fraction (CLAUDE.md generation rule: report per stage)
    n_cap = sum(
        1
        for per_ctx in results
        for text in per_ctx
        if len(tok(text, add_special_tokens=False)["input_ids"]) >= GEN_MAX_NEW_TOKENS
    )
    total = PLAN_COMPLETIONS
    total_completions = int(sum(total.values()))
    projected_gpu_h = total_completions * s_per / 3600.0
    pilot = {
        "behavior": behavior,
        "direction": "rb_unit",
        "layer": PILOT_LAYER,
        "position": "context",
        "dose_c": PILOT_DOSE_C,
        "alpha": float(alpha),
        "n_questions": len(contexts),
        "n_draws": args.pilot_draws,
        "n_completions": n_completions,
        "max_new_tokens": GEN_MAX_NEW_TOKENS,
        "wall_s": wall,
        "s_per_completion": s_per,
        "cap_hit_fraction": n_cap / max(n_completions, 1),
        "gpu": torch.cuda.get_device_name(0),
        "extrapolation": {
            "plan_completions": total,
            "total_completions": total_completions,
            "projected_gpu_h_at_pilot_rate": projected_gpu_h,
            "projected_wall_h_4gpu": projected_gpu_h / 4.0,
            "gate1_threshold_gpu_h": GATE1_THRESHOLD_GPU_H,
            "note": "gate-1 decision (trim rule / halt) is orchestrator-owned (plan §7)",
        },
    }
    logger.info(
        "[norm_probe] timing pilot: %.1fs wall / %d completions = %.2f s/completion "
        "-> projected %.1f GPU-h (threshold %.0f)",
        wall,
        n_completions,
        s_per,
        projected_gpu_h,
        GATE1_THRESHOLD_GPU_H,
    )
    return pilot


def _compute_rho(model, tok, behaviors: list[str], layers: list[int], phase: str = "norm_probe"):
    """Median last-context-token residual norm per (behavior, layer) + pooled.

    Returns ``(per_behavior, pooled_median)`` with ``"L<ly>"`` keys. Per-context
    SINGLE-ROW unpadded forwards (the exact #2220 geometry) so the parity /
    seam asserts compare like with like (gotchas.md matched batch geometry).
    """
    import torch

    from explore_persona_space.analysis.extraction import extract_layer_activations
    from explore_persona_space.experiments.issue1415 import steering

    result: dict[str, dict[str, float]] = {}
    pooled: dict[int, list[float]] = {ly: [] for ly in layers}
    t0 = time.time()
    for bi, behavior in enumerate(behaviors, 1):
        questions = _eval_questions(behavior)
        contexts = _contexts_for_questions(questions)
        norms: dict[int, list[float]] = {ly: [] for ly in layers}
        for ctx in contexts:
            ids = steering.context_token_ids(tok, ctx)
            input_ids = torch.tensor([ids], device=model.device)
            attn = torch.ones_like(input_ids)
            acts = extract_layer_activations(
                model, input_ids, layers, attention_mask=attn, detach_to_cpu=True
            )
            for ly in layers:
                vec = np.asarray(acts[ly][0, -1], dtype=np.float64)
                nrm = float(np.linalg.norm(vec))
                norms[ly].append(nrm)
                pooled[ly].append(nrm)
        result[behavior] = {f"L{ly}": float(np.median(norms[ly])) for ly in layers}
        _progress(phase, bi, len(behaviors), behavior, t0)
    pooled_median = {f"L{ly}": float(np.median(pooled[ly])) for ly in layers}
    return result, pooled_median


def phase_norm_probe(args) -> None:
    """rho_l = median last-context-token residual norm, all 28 layers, per
    behavior + pooled, with the #2220 shared-layer parity assert; then the
    1-cell timing pilot (plan §4.2 + §7 gate 1).

    Geometry note: per-context SINGLE-ROW unpadded forwards (the exact #2220
    phase_norm_probe geometry) so the parity assert compares like with like —
    a padded batched forward would shift bf16 numerics (gotchas.md matched
    batch geometry).

    Stated deviation (carried to the clean-result Methodology): the plan's
    flat parity rtol 5e-3 is split — 5e-3 stays binding for evil (parity
    inputs code-resident on both sides; the kernel/recipe canary) while the
    bank-loaded behaviors (hallucination, sycophancy) assert at
    RHO_PARITY_RTOL_BANKED=2e-2, because #2220's committed reference rho was
    computed on its pod's untracked bank cache (Sonnet-regen fallback) while
    this run stages the canonical sha-pinned #779 banks (events v20 diagnosis,
    2026-08-13; waived cells are recorded per-cell in rw2220_parity).
    """
    _require_cuda("norm_probe")

    out_root = _out_root(args)
    _assert_phase_headroom(out_root, 1.0, "norm_probe")
    layers = sorted(int(x) for x in args.layers)
    behaviors = list(args.behaviors)
    _breadcrumb("norm_probe", behaviors=len(behaviors), layers=len(layers))
    model, tok = _load_model_and_tokenizer()

    result, pooled_median = _compute_rho(model, tok, behaviors, layers, phase="norm_probe")

    parity = _rho_parity_assert(result)
    payload = _run_metadata(
        {
            "rho_median_last_context_token": result,
            "rho_pooled_median": pooled_median,
            "layers": layers,
            "rw2220_parity": parity,
        }
    )
    _write_json_atomic(out_root / "norm_probe" / "rho_by_layer.json", payload)

    pilot = _timing_pilot(args, model, tok, result)
    _write_json_atomic(out_root / "norm_probe" / "timing_pilot.json", _run_metadata(pilot))
    # rho is the pod-B seam input (decisive/patch/margin recompute + assert
    # against this file), so it must be fetchable off-pod (plan §4.2).
    _upload_folder_to_hf(out_root / "norm_probe", f"{_hf_prefix()}/norm_probe")

    _write_sentinel(
        out_root,
        "norm_probe",
        "done",
        {
            "s_per_completion": pilot["s_per_completion"],
            "projected_gpu_h": pilot["extrapolation"]["projected_gpu_h_at_pilot_rate"],
        },
    )
    _breadcrumb("norm_probe", status="done")


# ---------------------------------------------------------------------------
# unit-3 constants: steering grid + patch + judge/reduce (plan §4.2/§4.3)
# ---------------------------------------------------------------------------

MID_BAND = (14, 17, 20)
LAYER_CONFIGS = {
    "L14": (14,),
    "L17": (17,),
    "L20": (20,),
    "L26": (26,),
    "mid": MID_BAND,
    "all": ALL_LAYERS,
}
BREADTH_OF_CONFIG = {
    "L14": "single",
    "L17": "single",
    "L20": "single",
    "L26": "single",
    "mid": "mid",
    "all": "all",
}
BREADTHS = ("single", "mid", "all")
DOSES = (-4.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 4.0)
CONTEXT_DIRECTIONS = ("pre", "rb", "ctxext", "random", "preshuf")
ANSWER_DIRECTIONS = ("pre", "rb", "random")
NULL_STEER = {"context": ("random", "preshuf"), "answer": ("random",)}
PATCH_DIRECTIONS = ("pre", "rb", "ctxext")
PATCH_OPS = ("proj", "ablate")
PATCH_BREADTHS = ("single", "mid")
SEED_LOCALIZE = 42
SEEDS_DECISIVE = (42, 43)
JUDGE_DRAWS = {"localize": 3, "baseline_ceiling": 5, "decisive": 5, "patch": 5}
# multi-field #2220 trait rubric => the 2048 floor (llm-judging.md rule 23);
# the issue_1739 JUDGE_MAX_TOKENS=400 parity pin is NOT inherited here.
JUDGE_MAX_TOKENS_2254 = 2048
SCORE_THRESHOLD = 50.0
N_BOOT_CELL = 1000
N_BOOT_VERDICT = 2000
BOOTSTRAP_SEED = 20254
COHERENCE_CELL_GATE = 0.5
CAP_HIT_REGEN_FRAC = 0.02
CAP_HIT_REGEN_FACTOR = 2
POOL_SIZE = 10  # rw2220 margin pool size (verbatim reuse)
POOL_MIN = 3
MARGIN_BATCH_2254 = int(os.environ.get("EPM_MARGIN_BATCH", "4"))
RW2220_POOL_PREFIX = "issue2220_readwrite/margin/pools"
JUDGE_PILOT_MIN_EFFECTIVE = 51  # rule-26 floor at the 2% threshold (#2124)
_WAVE_SRC = {"localize": ("baseline_ceiling", "localize"), "decisive": ("decisive", "patch")}
_DIR_SHORT = {"pre": "pre", "rb": "rb", "ctxext": "cxd", "random": "rnd", "preshuf": "shf"}
_POS_SHORT = {"context": "ctx", "answer": "ans"}


def _c_token(c: float) -> str:
    """Dose token: -0.5 -> 'cm0p5', 2.0 -> 'c2'."""
    return "c" + ("m" if c < 0 else "") + f"{abs(c):g}".replace(".", "p")


def _cell_tokens(cell: dict) -> list[str]:
    """Stable id tokens for a generation cell (kind: alpha0|ceiling|steer|patch)."""
    b = cell["behavior"]
    kind = cell["kind"]
    if kind == "alpha0":
        return [b, "a0"]
    if kind == "ceiling":
        return [b, "cl"]
    if kind == "patch":
        return [
            b,
            _DIR_SHORT[cell["direction"]],
            "pp" if cell["op"] == "proj" else "ab",
            cell["breadth"],
        ]
    return [
        b,
        _DIR_SHORT[cell["direction"]],
        _POS_SHORT[cell["position"]],
        cell["layer_config"],
        _c_token(cell["c"]),
    ]


def _cell_id(cell: dict) -> str:
    return "__".join(_cell_tokens(cell))


def _judge_ctx_id(cell: dict, seed: int, i: int) -> str:
    """Judge context id: '-'-joined (rollout_item_id forbids '__'), <=49 chars."""
    cid = "-".join(_cell_tokens(cell)) + f"-s{seed}-x{i:03d}"
    assert len(cid) <= 49, cid
    return cid


# ---------------------------------------------------------------------------
# unit-3 loaders: direction bank, rho, pod-B rho seam
# ---------------------------------------------------------------------------


def _ensure_direction_vec(out_root: Path, behavior: str, slug: str, layer: int):
    """Unit-norm fp32 direction (H,) from the capture_directions bank
    (local-first, HF `stage_hub_file` fallback; fail-loud on shape)."""
    import torch

    from explore_persona_space.orchestrate import hub

    name = f"{behavior}_{slug}_L{layer}.pt"
    path = out_root / "directions" / name
    if not path.exists():
        hub.stage_hub_file(
            HF_DATA_REPO, f"{_hf_prefix()}/directions/{name}", path, repo_type="dataset"
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    vec = payload["direction"].float()
    assert vec.shape == (HIDDEN_DIM,), (name, tuple(vec.shape))
    return vec / vec.norm()


def _load_rho(out_root: Path) -> tuple[dict[str, float], dict]:
    """(pooled-median rho 'L<ly>'->float, full rho_by_layer payload);
    local-first, HF fallback (plan §4.2: dose rho = POOLED median)."""
    from explore_persona_space.orchestrate import hub

    path = out_root / "norm_probe" / "rho_by_layer.json"
    if not path.exists():
        hub.stage_hub_file(
            HF_DATA_REPO, f"{_hf_prefix()}/norm_probe/rho_by_layer.json", path, repo_type="dataset"
        )
    data = json.loads(path.read_text())
    pooled = {k: float(v) for k, v in data["rho_pooled_median"].items()}
    return pooled, data


def _rho_seam_assert(args, model, tok) -> dict:
    """Pod-B rho seam (plan §4.2, critic-mandated): recompute rho_l on THIS
    machine via the same `_compute_rho` and assert per-(behavior, layer)
    equality vs the staged pod-A rho_by_layer.json. Absence or mismatch is a
    hard fail — never a silent re-derive. Tolerance is per-behavior via
    ``_parity_rtol_for``, kept consistent with the pod-A #2220 parity gate;
    on THIS seam both sides share the #2254-staged sha-pinned banks, so
    bank-loaded cells are still expected within 5e-3 — any cell above 5e-3
    but within the banked tolerance is recorded loudly under
    ``provenance_waived_cells``, never silently equal-treated."""
    out_root = _out_root(args)
    _, data = _load_rho(out_root)
    ref = data["rho_median_last_context_token"]
    behaviors = sorted(ref)
    layers = sorted(int(k[1:]) for k in next(iter(ref.values())))
    fresh, _ = _compute_rho(model, tok, behaviors, layers, phase="rho_seam")
    n_checked = 0
    waived_cells: list[dict] = []
    for b in behaviors:
        rtol, banked = _parity_rtol_for(b)
        for key, ref_v in ref[b].items():
            new_v = fresh[b][key]
            rel = abs(new_v - float(ref_v)) / max(abs(float(ref_v)), 1e-9)
            if rel > rtol:
                raise RuntimeError(
                    f"rho seam mismatch: {b}/{key} fresh={new_v:.6g} vs staged={ref_v:.6g} "
                    f"rel={rel:.2e} > rtol={rtol:.0e} (banked={banked}) — pod-B "
                    "model/geometry drift vs pod-A"
                )
            if banked and rel > RHO_PARITY_RTOL:
                waived_cells.append(
                    {
                        "behavior": b,
                        "layer": key,
                        "fresh": new_v,
                        "staged": float(ref_v),
                        "rel_dev": rel,
                        "tolerance_applied": rtol,
                        "banked": banked,
                        "provenance_waived": True,
                    }
                )
            n_checked += 1
    assert n_checked > 0, "rho seam checked zero (behavior, layer) cells"
    logger.info(
        "[rho-seam] PASS: %d cells (rtol %s code-resident / %s bank-loaded; %d waived)",
        n_checked,
        RHO_PARITY_RTOL,
        RHO_PARITY_RTOL_BANKED,
        len(waived_cells),
    )
    return {
        "rho_seam_cells": n_checked,
        "rho_seam_rtol": RHO_PARITY_RTOL,
        "rho_seam_rtol_banked": RHO_PARITY_RTOL_BANKED,
        "provenance_waived_cells": waived_cells,
    }


# ---------------------------------------------------------------------------
# unit-3 hook factories + generation-grid driver
# ---------------------------------------------------------------------------


def _zero_hook_factory(model, layer: int, all_positions: bool = False):
    """(make, alphas) building a DeltaHook with alpha=0 — alpha0/ceiling cells
    run the IDENTICAL hook + generate path as steered cells (parity by
    construction, plan §4.2)."""
    import torch

    from explore_persona_space.experiments.issue1415.steering import DeltaHook

    delta = torch.zeros(HIDDEN_DIM, dtype=torch.bfloat16, device=model.device)

    def make():
        return DeltaHook(model, layer, delta, 0.0, all_positions=all_positions)

    return make, {f"L{layer}": 0.0}


def _steer_hook_factory(model, out_root: Path, cell: dict, rho_pooled: dict[str, float]):
    """(make, alphas) for a steer cell: alpha_l = (c/K) * rho_pooled[L_l]
    (plan §4.2 norm-match split; K = band width). position=answer steers
    every generated position (all_positions=True); context steers the
    last-context-token prefill slot only."""
    import torch

    from explore_persona_space.experiments.issue1415.steering import DeltaHook
    from explore_persona_space.experiments.issue2254.hooks import multi_layer_delta_hooks

    layers = list(LAYER_CONFIGS[cell["layer_config"]])
    c = float(cell["c"])
    k = len(layers)
    all_positions = cell["position"] == "answer"
    dirs = [
        _ensure_direction_vec(out_root, cell["behavior"], cell["direction"], ly).to(
            dtype=torch.bfloat16
        )
        for ly in layers
    ]
    alphas = [(c / k) * rho_pooled[f"L{ly}"] for ly in layers]
    if k == 1:

        def make():
            return DeltaHook(model, layers[0], dirs[0], alphas[0], all_positions=all_positions)

    else:

        def make():
            return multi_layer_delta_hooks(model, layers, dirs, alphas, all_positions=all_positions)

    return make, {f"L{ly}": a for ly, a in zip(layers, alphas, strict=True)}


def _cap_hit_fraction(res: list[list[str]], tok, cap: int) -> float:
    """Fraction of completions at/over the token cap (finish-reason proxy)."""
    n_hit = 0
    n_tot = 0
    for per_ctx in res:
        for t in per_ctx:
            n_tot += 1
            if len(tok(t, add_special_tokens=False)["input_ids"]) >= cap:
                n_hit += 1
    return n_hit / max(1, n_tot)


def _gen_cell_rows(
    model, tok, cell, contexts, q_of_context, hook_make, *, n_draws, seeds, max_new_tokens, alphas
):
    """Generate one cell: per seed, n_draws per context under the cell's hook;
    per-context coherence flags; cap-hit fraction over all draws. Content
    hygiene: completions land in the JSON payload only, never in logs."""
    from explore_persona_space.experiments.issue1415 import steering

    seeds_out: dict[str, dict] = {}
    cap_fracs = []
    for seed in seeds:
        with hook_make() as hook:
            res = steering.generate_batch(
                model,
                tok,
                contexts,
                n=n_draws,
                hook=hook,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                seed_base=int(seed),
            )
        coh = [steering.coherence_check(per_ctx) for per_ctx in res]
        seeds_out[str(seed)] = {
            "completions": res,
            "coherent_flags": coh,
            "condition_passes": [steering.condition_passes(flags) for flags in coh],
        }
        cap_fracs.append(_cap_hit_fraction(res, tok, max_new_tokens))
    return {
        "cell_id": _cell_id(cell),
        "cell": cell,
        "alphas": alphas,
        "q_of_context": q_of_context,
        "seeds": seeds_out,
        "max_new_tokens": max_new_tokens,
        "cap_hit_fraction": float(np.mean(cap_fracs)),
    }


def _run_gen_grid(
    args, phase, cells, *, contexts_of, q_of, hookf_builder, n_draws_of, seeds_of, pre_gen=None
):
    """Shared generation-grid driver: shard -> per-cell checkpoint JSONs under
    <out_root>/<phase>/raw_completions/ (cached-skip resume unless --force),
    cap-hit regen once at 2x cap (>2% trigger, CLAUDE.md max_new_tokens rule),
    HF raw-completions upload BEFORE the sentinel, per-cell progress lines."""
    _require_cuda(phase)
    out_root = _out_root(args)
    _assert_phase_headroom(out_root, 2.0, phase)
    if not cells:
        raise RuntimeError(f"{phase}: empty cell list (selection bug — never a silent no-op)")
    assert 0 <= args.shard_id < args.num_shards, (args.shard_id, args.num_shards)
    shard = cells[args.shard_id :: args.num_shards]
    comp_root = out_root / phase / "raw_completions"
    comp_root.mkdir(parents=True, exist_ok=True)
    _breadcrumb(phase, cells=len(cells), shard=len(shard), shard_id=args.shard_id)
    if not shard:
        # num_shards > len(cells): a legitimately EMPTY shard (baseline_ceiling
        # has 2 cells/behavior). Nothing generated => skip the folder upload
        # (other shards carry the cells; a bare upload of nothing would raise
        # the confusing "no files under ..." error — review minor g3). Reached
        # only with num_shards > 1: the empty-CELLS case above already raised.
        logger.warning(
            "[%s] shard %d/%d is EMPTY (%d cells < num_shards) — nothing to "
            "generate; skipping upload",
            phase,
            args.shard_id,
            args.num_shards,
            len(cells),
        )
        _write_sentinel(
            out_root,
            f"{phase}-shard{args.shard_id}",
            "done",
            {"cells": 0, "regen_cells": 0, "empty_shard": True},
        )
        _breadcrumb(phase, status="done", regen_cells=0, empty_shard=1)
        return
    model, tok = _load_model_and_tokenizer()
    sentinel_extra = dict(pre_gen(model, tok)) if pre_gen is not None else {}
    hookf = hookf_builder(model)
    t0 = time.time()
    n_regen = 0
    for k, cell in enumerate(shard, 1):
        cid = _cell_id(cell)
        path = comp_root / f"{cid}.json"
        if path.exists() and not args.force:
            _progress(phase, k, len(shard), f"{cid} (cached)", t0)
            continue
        contexts = contexts_of(cell)
        q_idx = q_of(cell)
        assert len(contexts) == len(q_idx), (cid, len(contexts), len(q_idx))
        make, alphas = hookf(cell)
        rec = _gen_cell_rows(
            model,
            tok,
            cell,
            contexts,
            q_idx,
            make,
            n_draws=n_draws_of(cell),
            seeds=seeds_of(cell),
            max_new_tokens=GEN_MAX_NEW_TOKENS,
            alphas=alphas,
        )
        if rec["cap_hit_fraction"] > CAP_HIT_REGEN_FRAC:
            n_regen += 1
            logger.info(
                "[%s] %s cap-hit %.3f > %.2f — regenerating at %dx cap",
                phase,
                cid,
                rec["cap_hit_fraction"],
                CAP_HIT_REGEN_FRAC,
                CAP_HIT_REGEN_FACTOR,
            )
            rec = _gen_cell_rows(
                model,
                tok,
                cell,
                contexts,
                q_idx,
                make,
                n_draws=n_draws_of(cell),
                seeds=seeds_of(cell),
                max_new_tokens=GEN_MAX_NEW_TOKENS * CAP_HIT_REGEN_FACTOR,
                alphas=alphas,
            )
        _write_json_atomic(path, _run_metadata(rec))
        _progress(phase, k, len(shard), cid, t0)
    _upload_folder_to_hf(comp_root, f"{_hf_prefix()}/raw_completions/{phase}")
    tag = phase if args.num_shards == 1 else f"{phase}-shard{args.shard_id}"
    _write_sentinel(
        out_root, tag, "done", {"cells": len(shard), "regen_cells": n_regen, **sentinel_extra}
    )
    _breadcrumb(phase, status="done", regen_cells=n_regen)


def _positive_instructions(behavior: str) -> list[str]:
    """The 5 POSITIVE extraction system prompts (donor-swap ceiling + patch
    persona prefixes; plan §4.2/§4.3). Prompt text is never logged."""
    _assert_e1_staged(behavior)  # regen fallback unreachable (blocker g2)
    from explore_persona_space.experiments.issue_1739.generation import load_e1_assets

    assets = load_e1_assets(behavior)
    return [p["pos"] for p in assets["instruction"][:N_INSTRUCTION_PAIRS]]


# ---------------------------------------------------------------------------
# unit-3 grid enumeration (smoke narrows COUNTS, never the code path)
# ---------------------------------------------------------------------------


def _grid_combos(args) -> list[tuple[str, str]]:
    """(direction, position) combos: 5 context + 3 answer = 8 (plan §4.2);
    smoke keeps one steered combo per position class plus a null direction."""
    if args.smoke:
        return [("pre", "context"), ("rb", "answer"), ("random", "context")]
    return [(d, "context") for d in CONTEXT_DIRECTIONS] + [(d, "answer") for d in ANSWER_DIRECTIONS]


def _grid_layer_configs(args) -> tuple[str, ...]:
    """Layer configs: 4 singles + mid band + all-28 (smoke: one single + mid)."""
    return ("L14", "mid") if args.smoke else tuple(LAYER_CONFIGS)


def _grid_doses(args) -> tuple[float, ...]:
    return (-0.5, 1.0) if args.smoke else DOSES


def _localize_cells(args, behaviors) -> list[dict]:
    """Localize grid: 8 combos x 6 layer-configs x 8 doses + alpha0 = 385
    cells/behavior (1,155 total at 3 behaviors; plan §4.2)."""
    cells: list[dict] = []
    for b in behaviors:
        n_before = len(cells)
        cells.append({"behavior": b, "kind": "alpha0"})
        for d, p in _grid_combos(args):
            for lc in _grid_layer_configs(args):
                for c in _grid_doses(args):
                    cells.append(
                        {
                            "behavior": b,
                            "kind": "steer",
                            "direction": d,
                            "position": p,
                            "layer_config": lc,
                            "c": float(c),
                        }
                    )
        if not args.smoke:
            assert len(cells) - n_before == 385, len(cells) - n_before
    return cells


def _load_reduce_json(out_root: Path, rel: str, hint: str) -> dict:
    """Local-first wave-1 reduce output; HF-staged fallback; fail-loud with a
    remediation hint when absent in both places (plan §4.2 pod-B gate)."""
    from explore_persona_space.orchestrate import hub

    path = out_root / rel
    if not path.exists():
        try:
            hub.stage_hub_file(HF_DATA_REPO, f"{_hf_prefix()}/{rel}", path, repo_type="dataset")
        except Exception as exc:
            raise FileNotFoundError(f"{rel} not found locally or on HF: {hint}") from exc
    return json.loads(path.read_text())


def _load_operating_points(out_root: Path) -> dict:
    return _load_reduce_json(
        out_root,
        "localize/operating_points.json",
        "run --phase judge_reduce --reduce-phase localize (wave 1) first",
    )


def _load_gates(out_root: Path) -> dict:
    return _load_reduce_json(
        out_root,
        "localize/gates.json",
        "run --phase judge_reduce --reduce-phase localize (wave 1) first",
    )


def _gate_ok_behaviors(gates: dict, behaviors: list[str]) -> tuple[list[str], list[str]]:
    """(gate-passing, skipped) behaviors per wave-1 gates.json verdicts."""
    kept: list[str] = []
    skipped: list[str] = []
    for b in behaviors:
        if b not in gates["behaviors"]:
            raise RuntimeError(f"gates.json missing behavior {b!r} — wave-1 reduce incomplete")
        (kept if gates["behaviors"][b]["proceed"] else skipped).append(b)
    return kept, skipped


# ---------------------------------------------------------------------------
# phase: baseline_ceiling (GPU; alpha0 baseline + donor-swap ceiling)
# ---------------------------------------------------------------------------


def phase_baseline_ceiling(args) -> None:
    """Alpha0 baseline (eval questions x 5 draws x seeds {42,43}) + donor-swap
    ceiling (5 positive extraction system prompts x eval questions, 1 draw x
    seeds {42,43}); zero-delta hooks keep the generate path identical to the
    steered cells (plan §4.2)."""
    _ensure_repo_root_on_syspath()
    import scripts.issue2220_readwrite as rw2220

    behaviors = list(args.behaviors)
    rw2220._assert_eval_bank_disjoint(behaviors)
    q_cache = {b: _eval_questions(b)[: args.q_decisive] for b in behaviors}
    pos_cache = {b: _positive_instructions(b) for b in behaviors}

    cells: list[dict] = []
    for b in behaviors:
        cells.append({"behavior": b, "kind": "alpha0"})
        cells.append({"behavior": b, "kind": "ceiling"})

    def contexts_of(cell):
        b = cell["behavior"]
        qs = q_cache[b]
        if cell["kind"] == "ceiling":
            return [{"system": instr, "user": q} for instr in pos_cache[b] for q in qs]
        return _contexts_for_questions(qs)

    def q_of(cell):
        b = cell["behavior"]
        nq = len(q_cache[b])
        if cell["kind"] == "ceiling":
            return [i % nq for i in range(len(pos_cache[b]) * nq)]
        return list(range(nq))

    def hookf_builder(model):
        def hookf(cell):
            return _zero_hook_factory(model, FROZEN_LAYER[cell["behavior"]])

        return hookf

    _run_gen_grid(
        args,
        "baseline_ceiling",
        cells,
        contexts_of=contexts_of,
        q_of=q_of,
        hookf_builder=hookf_builder,
        n_draws_of=lambda c: 1 if c["kind"] == "ceiling" else args.draws_decisive,
        seeds_of=lambda c: SEEDS_DECISIVE,
    )


# ---------------------------------------------------------------------------
# phase: localize (GPU; Q1 dose/layer/direction grid, pod A)
# ---------------------------------------------------------------------------


def phase_localize(args) -> None:
    """Q1 localization grid (plan §4.2): 8 direction x position combos x 6
    layer-configs x 8 doses + alpha0 per behavior; 10 questions x 3 draws x
    seed 42; per-cell JSON checkpoints (resume/shard-safe)."""
    out_root = _out_root(args)
    rho_pooled, _ = _load_rho(out_root)
    behaviors = list(args.behaviors)
    cells = _localize_cells(args, behaviors)
    q_cache = {b: _eval_questions(b)[: args.q_localize] for b in behaviors}

    def contexts_of(cell):
        return _contexts_for_questions(q_cache[cell["behavior"]])

    def q_of(cell):
        return list(range(len(q_cache[cell["behavior"]])))

    def hookf_builder(model):
        def hookf(cell):
            if cell["kind"] == "alpha0":
                return _zero_hook_factory(model, FROZEN_LAYER[cell["behavior"]])
            return _steer_hook_factory(model, out_root, cell, rho_pooled)

        return hookf

    _run_gen_grid(
        args,
        "localize",
        cells,
        contexts_of=contexts_of,
        q_of=q_of,
        hookf_builder=hookf_builder,
        n_draws_of=lambda c: args.draws_localize,
        seeds_of=lambda c: (SEED_LOCALIZE,),
    )


# ---------------------------------------------------------------------------
# phase: decisive (GPU, pod B; wave-1 operating points at full n)
# ---------------------------------------------------------------------------


def phase_decisive(args) -> None:
    """Q2 decisive grid at wave-1 operating points (plan §4.2): per
    gate-passing behavior, 8 combos x 3 breadths + alpha0; 20 questions x
    5 draws x seeds {42,43}. Pod-B seam: recompute rho and assert equality
    vs the staged pod-A values (hard fail on absence/mismatch)."""
    out_root = _out_root(args)
    ops = _load_operating_points(out_root)
    gates = _load_gates(out_root)
    kept, skipped = _gate_ok_behaviors(gates, list(args.behaviors))
    if not kept:
        raise RuntimeError(f"decisive: no gate-passing behaviors (skipped={skipped})")
    rho_pooled, _ = _load_rho(out_root)

    cells: list[dict] = []
    missing: list[str] = []
    for b in kept:
        cells.append({"behavior": b, "kind": "alpha0"})
        ops_b = ops["behaviors"][b]
        for d, p in _grid_combos(args):
            for breadth in BREADTHS:
                point = ops_b.get(f"{d}__{p}__{breadth}")
                if point is None:
                    missing.append(f"{b}/{d}/{p}/{breadth}")
                    continue
                cells.append(
                    {
                        "behavior": b,
                        "kind": "steer",
                        "direction": d,
                        "position": p,
                        "layer_config": point["layer_config"],
                        "c": float(point["c"]),
                    }
                )
    _write_json_atomic(
        out_root / "decisive" / "selection_meta.json",
        _run_metadata(
            {
                "skipped_behaviors": skipped,
                "missing_operating_points": missing,
                "gate_verdicts": {b: gates["behaviors"][b] for b in list(args.behaviors)},
            }
        ),
    )
    # Wave-2 completeness-gate input: the off-pod judge_reduce stages this via
    # _load_reduce_json (local-first, HF fallback) — upload BEFORE generation
    # starts, so a pod death mid-grid never strands the expected-set record.
    _upload_folder_to_hf(
        out_root / "decisive", f"{_hf_prefix()}/decisive", allow=["selection_meta.json"]
    )
    q_cache = {b: _eval_questions(b)[: args.q_decisive] for b in kept}

    def contexts_of(cell):
        return _contexts_for_questions(q_cache[cell["behavior"]])

    def q_of(cell):
        return list(range(len(q_cache[cell["behavior"]])))

    def hookf_builder(model):
        def hookf(cell):
            if cell["kind"] == "alpha0":
                return _zero_hook_factory(model, FROZEN_LAYER[cell["behavior"]])
            return _steer_hook_factory(model, out_root, cell, rho_pooled)

        return hookf

    _run_gen_grid(
        args,
        "decisive",
        cells,
        contexts_of=contexts_of,
        q_of=q_of,
        hookf_builder=hookf_builder,
        n_draws_of=lambda c: args.draws_decisive,
        seeds_of=lambda c: SEEDS_DECISIVE,
        pre_gen=lambda model, tok: _rho_seam_assert(args, model, tok),
    )


# ---------------------------------------------------------------------------
# phase: patch (GPU, pod B; projection-patch + directional ablation)
# ---------------------------------------------------------------------------


def _patch_layers_for(ops_b: dict, behavior: str, direction: str) -> list[int]:
    """Single-breadth patch layer for (behavior, direction): the wave-1
    single-breadth context operating point's layer; frozen-layer fallback when
    that combo produced no operating point."""
    point = ops_b.get(f"{direction}__context__single")
    if point is None:
        return [FROZEN_LAYER[behavior]]
    return list(LAYER_CONFIGS[point["layer_config"]])


def _patch_calibration(args, model, tok, out_root: Path, behaviors, ops) -> dict:
    """Calibration captures (plan §4.3): v_c_context over neutral + persona-
    prefixed contexts at every patch layer; per-context projections onto each
    direction persisted IN FULL plus the separation-vs-spread diagnostic."""
    import torch

    from explore_persona_space.experiments.issue1415 import steering

    calib: dict = {"behaviors": {}}
    for b in behaviors:
        qs = _eval_questions(b)[: args.q_decisive]
        neutral = _contexts_for_questions(qs)
        prefixed = [{"system": instr, "user": q} for instr in _positive_instructions(b) for q in qs]
        ops_b = ops["behaviors"][b]
        layer_set = sorted(
            {ly for d in PATCH_DIRECTIONS for ly in _patch_layers_for(ops_b, b, d)} | set(MID_BAND)
        )
        # Record (never silent — review minor g3) which directions fell back to
        # FROZEN_LAYER because their single-breadth context operating point is
        # missing (_patch_layers_for's fallback branch).
        fallback_dirs = sorted(
            d for d in PATCH_DIRECTIONS if ops_b.get(f"{d}__context__single") is None
        )
        if fallback_dirs:
            logger.warning(
                "[patch] %s: no single-breadth context operating point for %s — "
                "patch layer falls back to FROZEN_LAYER[%s]=%d (recorded in "
                "calibration_projections.json)",
                b,
                fallback_dirs,
                b,
                FROZEN_LAYER[b],
            )
        cap_neut = steering.capture_vectors(model, tok, neutral, layer_set)
        cap_pos = steering.capture_vectors(model, tok, prefixed, layer_set)
        li = {ly: i for i, ly in enumerate(layer_set)}
        rec_b: dict = {"layers": layer_set, "n_neutral": len(neutral), "n_pos": len(prefixed)}
        rec_b["frozen_layer_fallback_directions"] = fallback_dirs
        rec_b["directions"] = {}
        for d in PATCH_DIRECTIONS:
            per_layer = {}
            for ly in layer_set:
                dvec = _ensure_direction_vec(out_root, b, d, ly)
                pn = [
                    float(torch.as_tensor(r["v_c_context"])[li[ly]].float() @ dvec)
                    for r in cap_neut["per_context"]
                ]
                pp = [
                    float(torch.as_tensor(r["v_c_context"])[li[ly]].float() @ dvec)
                    for r in cap_pos["per_context"]
                ]
                sep = float(np.mean(pp) - np.mean(pn))
                per_layer[f"L{ly}"] = {
                    "proj_neutral": pn,
                    "proj_pos": pp,
                    "mu_neutral": float(np.mean(pn)),
                    "mu_pos": float(np.mean(pp)),
                    "sd_neutral": float(np.std(pn)),
                    "sd_pos": float(np.std(pp)),
                    "separation": sep,
                    "separation_over_spread": sep / float(np.std(pn) + 1e-12),
                }
            rec_b["directions"][d] = per_layer
        calib["behaviors"][b] = rec_b
    return calib


def phase_patch(args) -> None:
    """Q3 causal patch grid (plan §4.3): projection-patch on NEUTRAL contexts
    (<h,d> <- mu_pos, 20 ctx x 5 draws x 2 seeds) + directional ablation on
    persona-PREFIXED contexts (<h,d> <- mu_neutral, 100 ctx x 1 draw x 2
    seeds); 3 directions x 2 ops x 2 breadths per gate-passing behavior.
    The behavioral ceiling is REUSED from baseline_ceiling (same recipe)."""
    out_root = _out_root(args)
    ops = _load_operating_points(out_root)
    gates = _load_gates(out_root)
    kept, skipped = _gate_ok_behaviors(gates, list(args.behaviors))
    if not kept:
        raise RuntimeError(f"patch: no gate-passing behaviors (skipped={skipped})")

    q_cache = {b: _eval_questions(b)[: args.q_decisive] for b in kept}
    pos_cache = {b: _positive_instructions(b) for b in kept}
    cells = [
        {"behavior": b, "kind": "patch", "direction": d, "op": op_kind, "breadth": breadth}
        for b in kept
        for d in PATCH_DIRECTIONS
        for op_kind in PATCH_OPS
        for breadth in PATCH_BREADTHS
    ]
    calib_path = out_root / "patch" / "calibration_projections.json"

    def pre_gen(model, tok):
        seam = _rho_seam_assert(args, model, tok)
        if calib_path.exists() and not args.force:
            logger.info("[patch] calibration cached at %s", calib_path)
            return seam
        calib = _patch_calibration(args, model, tok, out_root, kept, ops)
        calib["rho_seam"] = seam
        _write_json_atomic(calib_path, _run_metadata(calib))
        _upload_folder_to_hf(
            out_root / "patch", f"{_hf_prefix()}/patch", allow=["calibration_projections.json"]
        )
        return seam

    def contexts_of(cell):
        b = cell["behavior"]
        if cell["op"] == "ablate":
            return [{"system": instr, "user": q} for instr in pos_cache[b] for q in q_cache[b]]
        return _contexts_for_questions(q_cache[b])

    def q_of(cell):
        b = cell["behavior"]
        nq = len(q_cache[b])
        if cell["op"] == "ablate":
            return [i % nq for i in range(len(pos_cache[b]) * nq)]
        return list(range(nq))

    def hookf_builder(model):
        calib = json.loads(calib_path.read_text())

        def hookf(cell):
            from explore_persona_space.experiments.issue2254.hooks import (
                ProjectionPatchHook,
                multi_layer_projection_patch_hooks,
            )

            b, d = cell["behavior"], cell["direction"]
            layers = (
                list(MID_BAND)
                if cell["breadth"] == "mid"
                else _patch_layers_for(ops["behaviors"][b], b, d)
            )
            key = "mu_pos" if cell["op"] == "proj" else "mu_neutral"
            dvecs = [_ensure_direction_vec(out_root, b, d, ly) for ly in layers]
            targets = [
                float(calib["behaviors"][b]["directions"][d][f"L{ly}"][key]) for ly in layers
            ]
            if len(layers) == 1:

                def make():
                    return ProjectionPatchHook(model, layers[0], dvecs[0], targets[0])

            else:

                def make():
                    return multi_layer_projection_patch_hooks(model, layers, dvecs, targets)

            return make, {f"L{ly}": t for ly, t in zip(layers, targets, strict=True)}

        return hookf

    _run_gen_grid(
        args,
        "patch",
        cells,
        contexts_of=contexts_of,
        q_of=q_of,
        hookf_builder=hookf_builder,
        n_draws_of=lambda c: args.draws_decisive if c["op"] == "proj" else 1,
        seeds_of=lambda c: SEEDS_DECISIVE,
        pre_gen=pre_gen,
    )


# ---------------------------------------------------------------------------
# phase: build_pools (VM CPU; margin answer pools)
# ---------------------------------------------------------------------------


def _build_pool_from_judged(out_root: Path, behavior: str) -> dict:
    """Regenerated pool (hallucination; plan §4.2): unique coherent judged
    localize completions -> pos = score>=50 top-POOL_SIZE / neg = score<50
    bottom-POOL_SIZE (the rw2220 pool-build recipe on this behavior's own
    judge-filtered completions, identical judge instrument)."""
    judged_dir = out_root / "judge" / "localize" / "judged"
    gen_dir = out_root / "localize" / "raw_completions"
    files = sorted(judged_dir.glob(f"{behavior}__*.json"))
    if not files:
        raise RuntimeError(
            f"build_pools({behavior}): no judged localize cells at {judged_dir} — "
            "run --phase judge_reduce --reduce-phase localize first"
        )
    best: dict[str, float] = {}
    for jf in files:
        jrec = json.loads(jf.read_text())
        grec = json.loads((gen_dir / jf.name).read_text())
        per_item = jrec["accounting"]["per_item_scores"]
        for iid, meta in jrec["items"].items():
            scores = per_item.get(iid) or []
            if not scores:
                continue
            sd = grec["seeds"][str(meta["seed"])]
            if not bool(sd["coherent_flags"][meta["ci"]][meta["di"]]):
                continue
            text = sd["completions"][meta["ci"]][meta["di"]]
            score = float(np.mean(scores))
            if text not in best or score > best[text]:
                best[text] = score
    uniq = sorted(best.items(), key=lambda kv: kv[1])
    neg = [t for t, s in uniq if s < SCORE_THRESHOLD][:POOL_SIZE]
    pos = [t for t, s in uniq if s >= SCORE_THRESHOLD][-POOL_SIZE:]
    if len(pos) < POOL_MIN or len(neg) < POOL_MIN:
        raise RuntimeError(
            f"build_pools({behavior}): pool below floor (pos={len(pos)}, neg={len(neg)}, "
            f"min={POOL_MIN}) — not enough judged coherent completions on both sides"
        )
    return {
        "pos": pos,
        "neg": neg,
        "provenance": {
            "source": "judged localize completions (issue 2254; identical judge instrument)",
            "n_unique_coherent": len(uniq),
            "recipe": "rw2220: dedup-max-score; neg = score<50 asc [:K]; pos = score>=50 [-K:]",
        },
    }


def phase_build_pools(args) -> None:
    """Margin answer pools (plan §4.2 secondary DV): evil + sycophancy staged
    VERBATIM from the rw2220 HF bank; hallucination REGENERATED from judged
    localize completions. VM/CPU-only; no GPU, no API calls."""
    out_root = _out_root(args)
    pools_dir = out_root / "pools"
    pools_dir.mkdir(parents=True, exist_ok=True)
    from explore_persona_space.orchestrate import hub

    for b in args.behaviors:
        target = pools_dir / f"{b}.json"
        if target.exists() and not args.force:
            logger.info("[build_pools] %s cached", b)
            continue
        if b in ("evil", "sycophancy"):
            hub.stage_hub_file(
                HF_DATA_REPO, f"{RW2220_POOL_PREFIX}/{b}.json", target, repo_type="dataset"
            )
            data = json.loads(target.read_text())
            assert len(data["pos"]) >= POOL_MIN and len(data["neg"]) >= POOL_MIN, (
                b,
                len(data["pos"]),
                len(data["neg"]),
            )
        else:
            _write_json_atomic(target, _run_metadata(_build_pool_from_judged(out_root, b)))
        _breadcrumb("build_pools", behavior=b, built=1)
    _upload_folder_to_hf(pools_dir, f"{_hf_prefix()}/pools")
    _write_sentinel(out_root, "build_pools", "done", {"behaviors": list(args.behaviors)})
    _breadcrumb("build_pools", status="done")


# ---------------------------------------------------------------------------
# phase: margin (GPU, pod B; teacher-forced fixed +/- pool margin)
# ---------------------------------------------------------------------------


def phase_margin(args) -> None:
    """Teacher-forced fixed positive-vs-negative pool margin at the decisive
    single-breadth operating points (plan §4.2 secondary DV; rw2220
    `_batched_ln_logp` verbatim reuse, batch 4). Per-cell checkpointing into
    margin/margin_percell.json (rewritten after EVERY cell); pod-B rho seam
    asserted before any scoring."""
    _ensure_repo_root_on_syspath()
    import scripts.issue2220_readwrite as rw2220
    from explore_persona_space.experiments.issue1415 import steering

    out_root = _out_root(args)
    ops = _load_operating_points(out_root)
    gates = _load_gates(out_root)
    kept, skipped = _gate_ok_behaviors(gates, list(args.behaviors))
    if not kept:
        raise RuntimeError(f"margin: no gate-passing behaviors (skipped={skipped})")
    pools: dict[str, dict] = {}
    for b in kept:
        pool_path = out_root / "pools" / f"{b}.json"
        if not pool_path.exists():
            raise FileNotFoundError(f"margin: missing pool {pool_path} — run --phase build_pools")
        pools[b] = json.loads(pool_path.read_text())

    _require_cuda("margin")
    _assert_phase_headroom(out_root, 1.0, "margin")
    model, tok = _load_model_and_tokenizer()
    seam = _rho_seam_assert(args, model, tok)
    rho_pooled, _ = _load_rho(out_root)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    assert pad_id is not None, "tokenizer exposes neither pad nor eos id"

    out_path = out_root / "margin" / "margin_percell.json"
    records: dict[str, dict] = {}
    if out_path.exists() and not args.force:
        records = json.loads(out_path.read_text()).get("cells", {})

    cells: list[dict] = []
    for b in kept:
        ops_b = ops["behaviors"][b]
        # alpha0 margin row passes a REAL direction tensor at alpha=0.0 —
        # rw2220._batched_ln_logp calls direction.to(...) unconditionally,
        # so None would crash; alpha=0 makes the hook a no-op numerically.
        cells.append(
            {
                "behavior": b,
                "direction": "rb",
                "position": "context",
                "layer": FROZEN_LAYER[b],
                "c": 0.0,
            }
        )
        for d, p in _grid_combos(args):
            point = ops_b.get(f"{d}__{p}__single")
            if point is None:
                continue
            cells.append(
                {
                    "behavior": b,
                    "direction": d,
                    "position": p,
                    "layer": int(LAYER_CONFIGS[point["layer_config"]][0]),
                    "c": float(point["c"]),
                }
            )
    if not cells:
        raise RuntimeError(f"margin: empty cell list (skipped={skipped})")

    t0 = time.time()
    for k, cell in enumerate(cells, 1):
        b = cell["behavior"]
        key = "__".join(
            [
                b,
                _DIR_SHORT[cell["direction"]],
                _POS_SHORT[cell["position"]],
                f"L{cell['layer']}",
                _c_token(cell["c"]),
                "mg",
            ]
        )
        if key in records and not args.force:
            _progress("margin", k, len(cells), f"{key} (cached)", t0)
            continue
        direction = _ensure_direction_vec(out_root, b, cell["direction"], cell["layer"])
        alpha = float(cell["c"]) * rho_pooled[f"L{cell['layer']}"]
        qs = _eval_questions(b)[: args.q_decisive]
        pos_ids = [tok.encode(a, add_special_tokens=False) for a in pools[b]["pos"]]
        neg_ids = [tok.encode(a, add_special_tokens=False) for a in pools[b]["neg"]]
        n_pos = len(pos_ids)
        per_context = []
        for ctx in _contexts_for_questions(qs):
            prompt_ids = steering.context_token_ids(tok, ctx)
            lps = rw2220._batched_ln_logp(
                model,
                prompt_ids,
                pos_ids + neg_ids,
                direction,
                cell["layer"],
                alpha,
                cell["position"],
                pad_id=pad_id,
                batch_size=MARGIN_BATCH_2254,
            )
            pos_lp = [x for x in lps[:n_pos] if np.isfinite(x)]
            neg_lp = [x for x in lps[n_pos:] if np.isfinite(x)]
            per_context.append(
                float(np.mean(pos_lp) - np.mean(neg_lp)) if pos_lp and neg_lp else float("nan")
            )
        records[key] = {
            "behavior": b,
            "direction": cell["direction"],
            "position": cell["position"],
            "layer": cell["layer"],
            "c": cell["c"],
            "alpha": alpha,
            "per_context_margin": per_context,
            "margin_mean": float(np.nanmean(per_context)),
            "n_pos": n_pos,
            "n_neg": len(neg_ids),
            "batch_size": MARGIN_BATCH_2254,
        }
        _write_json_atomic(
            out_path,
            _run_metadata({"cells": records, "rho_seam": seam, "skipped_behaviors": skipped}),
        )
        _progress("margin", k, len(cells), key, t0)
    _upload_folder_to_hf(out_root / "margin", f"{_hf_prefix()}/margin")
    _write_sentinel(out_root, "margin", "done", {"cells": len(records), **seam})
    _breadcrumb("margin", status="done")


# ---------------------------------------------------------------------------
# phase: judge_reduce (off-pod VM; Batch-API judging + wave reduces)
# ---------------------------------------------------------------------------


def _judge_draws(args, phase: str) -> int:
    return 2 if args.smoke else JUDGE_DRAWS[phase]


def _iter_gen_qa(rec: dict):
    """Yield (q_idx, seed, ctx_idx, draw_idx, text) rows of one gen cell."""
    for seed, sd in rec["seeds"].items():
        for ci, per_ctx in enumerate(sd["completions"]):
            qi = rec["q_of_context"][ci]
            for di, text in enumerate(per_ctx):
                yield qi, int(seed), ci, di, text


def _coherence_rate(rec: dict) -> float:
    """Fraction of contexts passing the generation-time coherence condition."""
    flags = [f for sd in rec["seeds"].values() for f in sd["condition_passes"]]
    return float(np.mean(flags)) if flags else 0.0


def _stage_phase_completions(out_root: Path, phase: str) -> Path:
    """Local-first raw_completions for <phase>; else stage every per-cell JSON
    from the HF prefix (scoped list_repo_tree via retry_transient — never a
    bare full-repo listing on the ~1M-file data repo)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    comp_root = out_root / phase / "raw_completions"
    if comp_root.exists() and any(comp_root.glob("*.json")):
        return comp_root
    prefix = f"{_hf_prefix()}/raw_completions/{phase}"
    entries = hub.retry_transient(
        lambda: list(
            HfApi().list_repo_tree(HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset")
        ),
        what=f"list_repo_tree({prefix})",
    )
    paths = [e.path for e in entries if e.path.endswith(".json")]
    if not paths:
        raise RuntimeError(f"judge_reduce: no completions for {phase} locally or at {prefix}")
    for pth in paths:
        hub.stage_hub_file(HF_DATA_REPO, pth, comp_root / Path(pth).name, repo_type="dataset")
    return comp_root


def _expected_gen_cell_ids(args, out_root: Path, wave: str) -> tuple[dict[str, set[str]], list]:
    """(expected gen-cell id set per source phase, pilot behaviors) for a wave.

    Wave 'localize': baseline_ceiling = {a0, cl} x behaviors; localize = the
    full `_localize_cells` enumeration (smoke-narrowed through args, the SAME
    enumeration the gen phase ran). Wave 'decisive': decisive = alpha0 + one
    cell per (combo x breadth) wave-1 operating point for gate-PASSING
    behaviors, honoring selection_meta.json's recorded
    `missing_operating_points`; patch = the 12-cell grid per kept behavior.
    """
    behaviors = list(args.behaviors)
    if wave == "localize":
        base = {
            _cell_id({"behavior": b, "kind": k}) for b in behaviors for k in ("alpha0", "ceiling")
        }
        loc = {_cell_id(c) for c in _localize_cells(args, behaviors)}
        return {"baseline_ceiling": base, "localize": loc}, behaviors
    ops = _load_operating_points(out_root)
    gates = _load_gates(out_root)
    kept, _skipped = _gate_ok_behaviors(gates, behaviors)
    meta = _load_reduce_json(
        out_root,
        "decisive/selection_meta.json",
        "run --phase decisive first (pod B writes + uploads it before generating)",
    )
    missing = set(meta.get("missing_operating_points", []))
    dec: set[str] = set()
    for b in kept:
        dec.add(_cell_id({"behavior": b, "kind": "alpha0"}))
        for d, p in _grid_combos(args):
            for breadth in BREADTHS:
                if f"{b}/{d}/{p}/{breadth}" in missing:
                    continue
                point = ops["behaviors"][b].get(f"{d}__{p}__{breadth}")
                if point is None:
                    raise RuntimeError(
                        f"judge_reduce: operating point {b}/{d}/{p}/{breadth} is null in "
                        "operating_points.json but NOT recorded in selection_meta.json "
                        "missing_operating_points — wave-1/decisive artifacts inconsistent"
                    )
                dec.add(
                    _cell_id(
                        {
                            "behavior": b,
                            "kind": "steer",
                            "direction": d,
                            "position": p,
                            "layer_config": point["layer_config"],
                            "c": float(point["c"]),
                        }
                    )
                )
    patch = {
        _cell_id({"behavior": b, "kind": "patch", "direction": d, "op": op_kind, "breadth": br})
        for b in kept
        for d in PATCH_DIRECTIONS
        for op_kind in PATCH_OPS
        for br in PATCH_BREADTHS
    }
    return {"decisive": dec, "patch": patch}, kept


def _assert_gen_grid_complete(args, out_root: Path, wave: str, comp_roots: dict) -> list:
    """Grid-completeness gate BEFORE the paid judge wave (review blocker g3,
    round 1): the staged gen-cell id set must COVER the deterministic expected
    grid — a partially-uploaded shard set or a partially-staged local dir must
    fail loud, never argmax operating points / populate null bands / decide
    gates over a partial grid (then spend pod B at the wrong points).

    Returns the pilot behavior list (wave localize: args.behaviors; wave
    decisive: the gate-PASSING behaviors only — a gate-demoted behavior has
    no decisive gen cells by design, so piloting it would false-crash)."""
    expected, pilot_behaviors = _expected_gen_cell_ids(args, out_root, wave)
    problems = []
    for phase, exp in expected.items():
        staged = {f.stem for f in comp_roots[phase].glob("*.json")}
        missing = sorted(exp - staged)
        if missing:
            problems.append(f"{phase}: {len(missing)} missing of {len(exp)} (e.g. {missing[:8]})")
    if problems:
        raise RuntimeError(
            "judge_reduce: staged gen grid INCOMPLETE — refusing to judge/reduce a "
            "partial cell set (some shards not yet generated/uploaded, or a partial "
            "local raw_completions/ dir shadowing a complete HF prefix — remove it "
            "to re-stage): " + "; ".join(problems)
        )
    return pilot_behaviors


def _run_judge_pilot(args, out_root: Path, gen_phase: str, behavior: str, rubric: str, n_draws):
    """Rule-26 pilot gate, ONE judge_pilot_gate call per behavior per wave
    (>=51 effective draws per arm at the 2% threshold; truncation FAIL
    unwaivable). Fingerprint sidecar skips a prior PASS at the identical
    instrument (rubric + n_draws + max_tokens) unless --force.

    Draw-count note (review minor g3): the pilot runs at the STEERED phase's
    draw count (wave 1: localize's 3) while the same wave also judges
    baseline_ceiling at 5 draws — rubric + max_tokens are identical, so the
    truncation/parse coverage carries; the mismatch is recorded in the pass
    sidecar."""
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate

    pilot_dir = out_root / "judge" / "pilot" / gen_phase
    pilot_dir.mkdir(parents=True, exist_ok=True)
    fp = _sha8(
        {"behavior": behavior, "rubric": rubric, "n_draws": n_draws, "mt": JUDGE_MAX_TOKENS_2254}
    )
    pass_path = pilot_dir / f"{behavior}.pass.json"
    if pass_path.exists() and not args.force:
        if json.loads(pass_path.read_text()).get("fingerprint") == fp:
            logger.info(
                "[judge-pilot] %s/%s: prior PASS, identical instrument", gen_phase, behavior
            )
            return
    comp_root = out_root / gen_phase / "raw_completions"
    files = sorted(comp_root.glob(f"{behavior}__*.json"))
    if not files:
        raise RuntimeError(f"judge pilot: no {behavior} gen cells under {comp_root}")
    items_per_arm = -(-JUDGE_PILOT_MIN_EFFECTIVE // n_draws)  # ceil
    qs = _eval_questions(behavior)

    def _collect(arm: str, cell_filter, sort_key) -> list[tuple[str, str, str]]:
        recs = []
        for f in files:
            rec = json.loads(f.read_text())
            if rec["cell"]["kind"] != "alpha0" and not cell_filter(rec["cell"]):
                continue
            recs.append((sort_key(rec), rec))
        recs.sort(key=lambda kv: kv[0])
        items: list[tuple[str, str, str]] = []
        for _k, rec in recs:
            for qi, _seed, _ci, _di, text in _iter_gen_qa(rec):
                items.append((f"{arm}-{len(items):03d}", qs[qi], text))
                if len(items) >= items_per_arm:
                    return items
        return items

    arms = {
        "ctx_steer": _collect(
            "ctx_steer",
            lambda c: c.get("position") == "context",
            lambda r: -abs(float(r["cell"].get("c", 0.0))),
        ),
        "ans_steer": _collect(
            "ans_steer",
            lambda c: c.get("position") == "answer",
            lambda r: -abs(float(r["cell"].get("c", 0.0))),
        ),
        "degen": _collect("degen", lambda c: True, _coherence_rate),
    }
    arms = {k: v for k, v in arms.items() if v}
    if not arms:
        raise RuntimeError(f"judge pilot: zero pilot items for {gen_phase}/{behavior}")
    report = judge_pilot_gate(
        arms,
        rubric,
        max_tokens=JUDGE_MAX_TOKENS_2254,
        cache_dir=pilot_dir / f"{behavior}_cache",
        save_raw_dir=pilot_dir / f"{behavior}_raw",
        n_draws=n_draws,
        target_total_draws=len(arms) * n_draws * items_per_arm,
        min_effective_draws_per_arm=JUDGE_PILOT_MIN_EFFECTIVE,
        allow_subresolution_pilot=bool(args.smoke),
        report_path=pilot_dir / f"{behavior}.report.json",
    )
    if not report.passed:
        raise RuntimeError(f"judge pilot FAILED for {gen_phase}/{behavior}: {report.verdict}")
    _write_json_atomic(
        pass_path,
        {
            "fingerprint": fp,
            "verdict": report.verdict,
            "draw_count_note": (
                f"pilot ran at the steered phase's n_draws={n_draws}; the same wave "
                "judges baseline_ceiling at 5 draws (rubric + max_tokens identical, "
                "truncation coverage carries)"
            ),
        },
    )


def _judge_cell(args, out_root: Path, phase: str, gen_path: Path, rubric: str, n_draws: int):
    """Judge one gen cell via judge_items_graded (Batch API, rubric-keyed
    cache, generous max_tokens); per-cell checkpoint at
    judge/<phase>/judged/<cid>.json with cached-skip resume. Accounting keeps
    the rule-9/24/28 drop-class split + rule-29 frac_items_complete."""
    from explore_persona_space.experiments.issue_1739.constants import (
        JUDGE_MODEL,
        JUDGE_TEMPERATURE,
    )
    from explore_persona_space.experiments.issue_1739.judging import (
        judge_items_graded,
        judge_tallies,
        rollout_item_id,
    )

    rec = json.loads(gen_path.read_text())
    cell = rec["cell"]
    cid = rec["cell_id"]
    out_path = out_root / "judge" / phase / "judged" / f"{cid}.json"
    if out_path.exists() and not args.force:
        return json.loads(out_path.read_text())
    qs = _eval_questions(cell["behavior"])
    items: list[tuple[str, str, str]] = []
    meta: dict[str, dict] = {}
    for qi, seed, ci, di, text in _iter_gen_qa(rec):
        iid = rollout_item_id(_judge_ctx_id(cell, seed, len(items)), di)
        items.append((iid, qs[qi], text))
        meta[iid] = {"qi": qi, "seed": seed, "ci": ci, "di": di}
    result = judge_items_graded(
        items,
        rubric,
        cache_dir=out_root / "judge" / phase / "cache" / cid,
        save_raw=out_root / "judge" / phase / "raw" / cid,
        n_draws=n_draws,
        temperature=JUDGE_TEMPERATURE,
        max_tokens=JUDGE_MAX_TOKENS_2254,
        judge_model=JUDGE_MODEL,
    )
    per_q: dict[int, list[float]] = {}
    for iid, scores in result.per_item_scores.items():
        if scores:
            per_q.setdefault(meta[iid]["qi"], []).append(float(np.mean(scores)))
    n_q = (max(m["qi"] for m in meta.values()) + 1) if meta else 0
    per_q_mean = [float(np.mean(per_q[q])) if q in per_q else None for q in range(n_q)]
    per_q_rate = [
        float(np.mean([s >= SCORE_THRESHOLD for s in per_q[q]])) if q in per_q else None
        for q in range(n_q)
    ]
    valid_means = [m for m in per_q_mean if m is not None]
    valid_rates = [r for r in per_q_rate if r is not None]
    coherence_rate = _coherence_rate(rec)
    out = {
        "cell_id": cid,
        "cell": cell,
        "phase": phase,
        "n_questions": n_q,
        "judge": {
            "model": JUDGE_MODEL,
            "n_draws": n_draws,
            "max_tokens": JUDGE_MAX_TOKENS_2254,
            "temperature": JUDGE_TEMPERATURE,
        },
        "items": meta,
        "accounting": {
            **judge_tallies(result),
            "n_refusal_draws": result.n_refusal_draws,
            "n_api_refusal_draws": result.n_api_refusal_draws,
            "per_item_api_refusals": result.per_item_api_refusals,
            "frac_items_complete": (result.frac_items_complete if result.scores else None),
            "n_items": len(result.scores),
            "n_items_zero_valid": sum(1 for s in result.per_item_scores.values() if not s),
        },
        "per_question_mean_score": per_q_mean,
        "per_question_rate": per_q_rate,
        "per_question_n": [len(per_q.get(q, [])) for q in range(n_q)],
        "mean_score": float(np.mean(valid_means)) if valid_means else None,
        "rate": float(np.mean(valid_rates)) if valid_rates else None,
        "coherence_rate": coherence_rate,
        "coherence_pass": bool(coherence_rate >= COHERENCE_CELL_GATE),
        "cap_hit_fraction": rec.get("cap_hit_fraction"),
        "alphas": rec.get("alphas"),
    }
    _write_json_atomic(out_path, _run_metadata(out))
    return out


# ---------------------------------------------------------------------------
# reduce: vectorized question-level paired cluster bootstrap + verdicts
# ---------------------------------------------------------------------------


def _boot_idx(nq: int, n_draws: int, seed_key: str) -> np.ndarray:
    """Deterministic (n_draws, nq) resample-index matrix; per-cell seed =
    BOOTSTRAP_SEED + crc32(cell key) % 100000 (plan §6)."""
    import zlib

    rng = np.random.default_rng(BOOTSTRAP_SEED + zlib.crc32(seed_key.encode()) % 100000)
    return rng.integers(0, nq, size=(n_draws, nq))


def _q_arr(judged: dict) -> np.ndarray:
    """Per-question mean-score vector (NaN where every draw dropped)."""
    return np.array(
        [np.nan if v is None else float(v) for v in judged["per_question_mean_score"]],
        dtype=np.float64,
    )


def _boot_diff_ci(cell_q: np.ndarray, ref_q: np.ndarray, idx: np.ndarray):
    """Question-level PAIRED cluster bootstrap of mean(cell) - mean(ref):
    identical resample indices on both arms, vectorized fancy-index."""
    diffs = np.nanmean(cell_q[idx], axis=1) - np.nanmean(ref_q[idx], axis=1)
    point = float(np.nanmean(cell_q) - np.nanmean(ref_q))
    return point, float(np.nanquantile(diffs, 0.025)), float(np.nanquantile(diffs, 0.975))


def _null_band(null_qarrs, a0_q: np.ndarray, seed_key: str, n_draws: int = N_BOOT_CELL):
    """Selection-symmetric null band (plan §6): per bootstrap draw, nanmax of
    the coherence-gated null cells' paired deltas (max-over-cells INSIDE each
    draw, so observed and null get identical selection chances)."""
    if not null_qarrs:
        return None
    idx = _boot_idx(len(a0_q), n_draws, seed_key)
    a0_b = np.nanmean(a0_q[idx], axis=1)
    per_cell = np.stack([np.nanmean(q[idx], axis=1) - a0_b for q in null_qarrs], axis=1)
    maxes = np.nanmax(per_cell, axis=1)
    return {
        "p50": float(np.nanquantile(maxes, 0.5)),
        "p975": float(np.nanquantile(maxes, 0.975)),
        "n_cells": int(len(null_qarrs)),
        "n_draws": int(n_draws),
    }


def _completeness_block(files, floor: float = 0.95) -> dict:
    """Rule-29 per-cell frac_items_complete vs the pre-registered floor."""
    rows: dict[str, float | None] = {}
    below: list[str] = []
    for f in files:
        j = json.loads(f.read_text())
        fc = j["accounting"]["frac_items_complete"]
        rows[j["cell_id"]] = fc
        if fc is not None and fc < floor:
            below.append(j["cell_id"])
    return {
        "floor": floor,
        "per_cell": rows,
        "below_floor_cells": below,
        "remediation": (
            "below-floor cells: triage drop class per llm-judging.md rule 29; "
            "api-refusal class -> targeted sync re-issue at the identical instrument "
            "(scripts/issue1739_evilood_refusal_rejudge.py recipe)"
        ),
    }


def _reduce_wave1(args, out_root: Path) -> None:
    """Wave-1 reduce (plan §4.2): per-cell dose-response deltas (frozen CIs),
    coherence-gated operating-point argmax per (direction, position, breadth),
    selection-symmetric null bands, gate-2/gate-3 verdicts, baseline/ceiling
    summary, rule-29 completeness + HF upload of the pod-B input JSONs."""
    jl = out_root / "judge" / "localize" / "judged"
    jb = out_root / "judge" / "baseline_ceiling" / "judged"
    dose: dict = {"behaviors": {}}
    op_out: dict = {"behaviors": {}}
    gates: dict = {"behaviors": {}}
    base_out: dict = {"behaviors": {}}
    for b in list(args.behaviors):
        a0 = json.loads((jl / f"{b}__a0.json").read_text())
        a0_q = _q_arr(a0)
        steer = []
        for f in sorted(jl.glob(f"{b}__*.json")):
            j = json.loads(f.read_text())
            if j["cell"]["kind"] == "steer":
                steer.append(j)
        if not steer:
            raise RuntimeError(f"reduce wave1: no steered localize cells for {b}")
        percell: dict = {}
        qarrs: dict = {}
        for j in steer:
            cid = j["cell_id"]
            cq = _q_arr(j)
            qarrs[cid] = cq
            point, lo, hi = _boot_diff_ci(cq, a0_q, _boot_idx(len(a0_q), N_BOOT_CELL, cid))
            percell[cid] = {
                "cell": j["cell"],
                "delta_score": point,
                "ci_frozen": [lo, hi],
                "ci_label": f"frozen (localize grid, n_q={len(a0_q)})",
                "delta_rate": (
                    None
                    if j["rate"] is None or a0["rate"] is None
                    else float(j["rate"] - a0["rate"])
                ),
                "mean_score": j["mean_score"],
                "coherence_rate": j["coherence_rate"],
                "coherence_pass": j["coherence_pass"],
                "frac_items_complete": j["accounting"]["frac_items_complete"],
                "cap_hit_fraction": j.get("cap_hit_fraction"),
            }

        def _null_cells(pos, steer=steer, qarrs=qarrs):
            return [
                qarrs[j["cell_id"]]
                for j in steer
                if j["cell"]["position"] == pos
                and j["cell"]["direction"] in NULL_STEER[pos]
                and j["coherence_pass"]
            ]

        band_ctx = _null_band(_null_cells("context"), a0_q, f"{b}__nullctx")
        band_ans = _null_band(_null_cells("answer"), a0_q, f"{b}__nullans")

        ops_b: dict = {}
        combos = [(d, "context") for d in CONTEXT_DIRECTIONS] + [
            (d, "answer") for d in ANSWER_DIRECTIONS
        ]
        for d, p in combos:
            for breadth in BREADTHS:
                cands = [
                    j
                    for j in steer
                    if j["cell"]["direction"] == d
                    and j["cell"]["position"] == p
                    and BREADTH_OF_CONFIG[j["cell"]["layer_config"]] == breadth
                    and j["coherence_pass"]
                ]
                if not cands:
                    ops_b[f"{d}__{p}__{breadth}"] = None
                    continue
                bestj = max(cands, key=lambda j: percell[j["cell_id"]]["delta_score"])
                ops_b[f"{d}__{p}__{breadth}"] = {
                    "cell_id": bestj["cell_id"],
                    "layer_config": bestj["cell"]["layer_config"],
                    "c": bestj["cell"]["c"],
                    "delta_score": percell[bestj["cell_id"]]["delta_score"],
                }

        base_a0 = json.loads((jb / f"{b}__a0.json").read_text())
        base_cl = json.loads((jb / f"{b}__cl.json").read_text())
        b_a0q, b_clq = _q_arr(base_a0), _q_arr(base_cl)
        ceil_pt, ceil_lo, ceil_hi = _boot_diff_ci(
            b_clq, b_a0q, _boot_idx(len(b_a0q), N_BOOT_CELL, f"{b}__ceiling")
        )
        rb_ans = [
            percell[j["cell_id"]]["delta_score"]
            for j in steer
            if j["cell"]["direction"] == "rb"
            and j["cell"]["position"] == "answer"
            and j["coherence_pass"]
        ]
        gate2 = {
            "pass": bool(rb_ans and band_ans is not None and max(rb_ans) > band_ans["p975"]),
            "best_rb_answer_delta": max(rb_ans) if rb_ans else None,
            "answer_band_p975": band_ans["p975"] if band_ans else None,
        }
        # Gate 3 — the REGISTERED quantity (plan §7 gate 3 + §6 band-vs-
        # ceiling): achievable ceiling on the primary scale = 100 − α0 mean
        # graded score (headroom) vs the context null-band upper edge on
        # Δscore; rate version (1 − α0 rate) reported alongside. The donor-
        # swap ceiling delta is kept as reported CONTEXT (it is the §4.3
        # patch-fraction denominator + the hero-2 scale), never the gate
        # criterion (review blocker g3, round 1).
        headroom = 100.0 - float(np.nanmean(b_a0q))
        headroom_rate = None if base_a0["rate"] is None else 1.0 - float(base_a0["rate"])
        gate3 = {
            "pass": bool(band_ctx is not None and headroom > band_ctx["p975"]),
            "headroom_score": headroom,
            "headroom_rate": headroom_rate,
            "ceiling_delta": ceil_pt,
            "ceiling_ci": [ceil_lo, ceil_hi],
            "baseline_mean": float(np.nanmean(b_a0q)),
            "ceiling_mean": float(np.nanmean(b_clq)),
            "context_band_p975": band_ctx["p975"] if band_ctx else None,
        }
        gates["behaviors"][b] = {
            "gate2": gate2,
            "gate3": gate3,
            "proceed": bool(gate2["pass"] and gate3["pass"]),
        }
        dose["behaviors"][b] = {
            "alpha0_mean": float(np.nanmean(a0_q)),
            "alpha0_rate": a0["rate"],
            "n_q": int(len(a0_q)),
            "cells": percell,
            "null_band_context": band_ctx,
            "null_band_answer": band_ans,
        }
        op_out["behaviors"][b] = ops_b
        keep = ("mean_score", "rate", "per_question_mean_score", "coherence_rate")
        base_out["behaviors"][b] = {
            "alpha0": {k: base_a0[k] for k in keep},
            "ceiling": {k: base_cl[k] for k in keep},
            "headroom_score": headroom,  # achievable ceiling, 100 − α0 mean (plan §6)
            "ceiling_delta": ceil_pt,
            "ceiling_ci": [ceil_lo, ceil_hi],
        }
    # Stated deviation (carried to the clean-result scope note): coherence
    # gating uses the programmatic steering.coherence_check/condition_passes
    # only — the judged 0-100 coherence covariate named in plan §6's
    # evaluation table is not collected, mirroring #2220's recorded deviation
    # (issue2220_readwrite.py L29-38; the plan pins "the #2220 instrument").
    gates["stated_deviation_coherence"] = (
        "coherence gate = programmatic steering.coherence_check/condition_passes; "
        "the judged 0-100 coherence covariate is NOT collected (mirrors #2220's "
        "recorded deviation, issue2220_readwrite.py L29-38) — carry to the "
        "clean-result scope note"
    )
    _write_json_atomic(out_root / "localize" / "dose_response.json", _run_metadata(dose))
    _write_json_atomic(out_root / "localize" / "operating_points.json", _run_metadata(op_out))
    _write_json_atomic(out_root / "localize" / "gates.json", _run_metadata(gates))
    _write_json_atomic(
        out_root / "baseline_ceiling" / "judged_percell.json", _run_metadata(base_out)
    )
    files = sorted(jl.glob("*.json")) + sorted(jb.glob("*.json"))
    _write_json_atomic(
        out_root / "judge" / "completeness_wave1.json", _run_metadata(_completeness_block(files))
    )
    # pod-B staging source: decisive/patch/margin fetch these three via
    # _load_reduce_json — upload BEFORE the sentinel.
    _upload_folder_to_hf(
        out_root / "localize",
        f"{_hf_prefix()}/localize",
        allow=["dose_response.json", "operating_points.json", "gates.json"],
    )
    _upload_folder_to_hf(
        out_root / "baseline_ceiling",
        f"{_hf_prefix()}/baseline_ceiling",
        allow=["judged_percell.json"],
    )


def _selection_inherited_for(out_root: Path, b: str, d: str, p: str):
    """Selection-inherited CI (selection-symmetric-nulls rule): per bootstrap
    draw, re-argmax over the (direction, position) LOCALIZE-grid cells;
    labeled with its own grid + n_q (the frozen decisive CI is co-reported)."""
    jl = out_root / "judge" / "localize" / "judged"
    a0_path = jl / f"{b}__a0.json"
    if not a0_path.exists():
        return None
    a0_q = _q_arr(json.loads(a0_path.read_text()))
    cells = []
    for f in sorted(jl.glob(f"{b}__{_DIR_SHORT[d]}__{_POS_SHORT[p]}__*.json")):
        j = json.loads(f.read_text())
        if j["coherence_pass"]:
            cells.append(_q_arr(j))
    if not cells:
        return None
    idx = _boot_idx(len(a0_q), N_BOOT_VERDICT, f"{b}__{d}__{p}__selinh")
    a0_b = np.nanmean(a0_q[idx], axis=1)
    per_cell = np.stack([np.nanmean(q[idx], axis=1) - a0_b for q in cells], axis=1)
    maxes = np.nanmax(per_cell, axis=1)
    return {
        "ci": [float(np.nanquantile(maxes, 0.025)), float(np.nanquantile(maxes, 0.975))],
        "label": f"selection_inherited (localize grid, n_q={len(a0_q)}, n_cells={len(cells)})",
    }


def _lattice_label(margins: dict) -> tuple[str, str]:
    """H1/H2/H3/Ambiguous/Undefined from the E_pre/E_ctxdir/C_gap lattice
    (plan §6). CIs here are band-shifted frozen CIs (E_*) and the paired
    2000-draw bootstrap (C_gap)."""
    e_pre = margins.get("E_pre")
    e_ctx = margins.get("E_ctxdir")
    gap = margins.get("C_gap")
    if e_pre is None or e_ctx is None or gap is None:
        return "Undefined", "missing operating-point cells (pre/ctxext @ context)"
    pre_pos = e_pre["ci"][0] > 0
    gap_neg = gap["ci"][1] < 0
    ctx_pos = e_ctx["ci"][0] > 0
    if pre_pos and not gap_neg:
        return "H1", "pre-image steers (CI>0 vs null band) and is not CI-below ctxext"
    if pre_pos and gap_neg:
        return "H3", "pre-image steers but sits CI-below the fitted-map direction"
    if not pre_pos and ctx_pos:
        return "H2", "pre-image does not clear the null band while ctxext does"
    return "Ambiguous", "neither margin resolves at the decisive grain"


def _patch_vs_ceiling(out_root: Path) -> dict:
    """Patch/ablation effects as a fraction of the REUSED baseline_ceiling
    donor-swap ceiling; per-draw denominators |.|<=1e-6 -> NaN with the
    degenerate-draw count reported (never a silent divide)."""
    jp = out_root / "judge" / "patch" / "judged"
    jb = out_root / "judge" / "baseline_ceiling" / "judged"
    out: dict = {"cells": {}}
    for f in sorted(jp.glob("*.json")):
        j = json.loads(f.read_text())
        b = j["cell"]["behavior"]
        a0_q = _q_arr(json.loads((jb / f"{b}__a0.json").read_text()))
        cl_q = _q_arr(json.loads((jb / f"{b}__cl.json").read_text()))
        cq = _q_arr(j)
        nq = min(len(cq), len(a0_q), len(cl_q))
        idx = _boot_idx(nq, N_BOOT_CELL, j["cell_id"] + "__pvc")
        cell_b = np.nanmean(cq[:nq][idx], axis=1)
        a0_b = np.nanmean(a0_q[:nq][idx], axis=1)
        cl_b = np.nanmean(cl_q[:nq][idx], axis=1)
        denom = cl_b - a0_b
        ok = np.abs(denom) > 1e-6
        if j["cell"]["op"] == "proj":
            frac = np.where(ok, (cell_b - a0_b) / np.where(ok, denom, 1.0), np.nan)
            point_num = float(np.nanmean(cq[:nq]) - np.nanmean(a0_q[:nq]))
        else:
            frac = np.where(ok, (cl_b - cell_b) / np.where(ok, denom, 1.0), np.nan)
            point_num = float(np.nanmean(cl_q[:nq]) - np.nanmean(cq[:nq]))
        point_den = float(np.nanmean(cl_q[:nq]) - np.nanmean(a0_q[:nq]))

        def _ci_edge(x) -> float | None:
            # strict-JSON: an all-degenerate cell's nanquantile is NaN and
            # json.dumps would emit a non-strict `NaN` literal (review minor
            # g3) — map non-finite edges to null.
            return float(x) if np.isfinite(x) else None

        out["cells"][j["cell_id"]] = {
            "cell": j["cell"],
            "fraction_point": (point_num / point_den) if abs(point_den) > 1e-6 else None,
            "fraction_ci": [
                _ci_edge(np.nanquantile(frac, 0.025)),
                _ci_edge(np.nanquantile(frac, 0.975)),
            ],
            "n_degenerate_draws": int((~ok).sum()),
            "mean_score": j["mean_score"],
            "coherence_rate": j["coherence_rate"],
            "coherence_pass": j["coherence_pass"],
        }
    return out


def _reduce_wave2(args, out_root: Path) -> None:
    """Wave-2 reduce (plan §4.2/§6): decisive per-cell deltas (frozen CIs) vs
    the decisive alpha0, per-behavior E_pre/E_ctxdir/C_gap margins against the
    decisive context null band, BOTH CI labels (frozen + selection-inherited
    re-argmax on the localize grid), lattice labels, patch-vs-ceiling."""
    jd = out_root / "judge" / "decisive" / "judged"
    gates = _load_gates(out_root)
    percell_out: dict = {"behaviors": {}}
    verdicts: dict = {"behaviors": {}}
    for b in list(args.behaviors):
        if not gates["behaviors"].get(b, {}).get("proceed"):
            verdicts["behaviors"][b] = {
                "label": "Undefined",
                "reason": "behavior skipped at wave-1 gates (gate 2/3)",
            }
            continue
        a0_path = jd / f"{b}__a0.json"
        if not a0_path.exists():
            raise FileNotFoundError(f"reduce wave2: missing decisive alpha0 judged cell for {b}")
        a0_q = _q_arr(json.loads(a0_path.read_text()))
        steer = []
        for f in sorted(jd.glob(f"{b}__*.json")):
            j = json.loads(f.read_text())
            if j["cell"]["kind"] == "steer":
                steer.append(j)
        if not steer:
            raise RuntimeError(f"reduce wave2: no steered decisive cells for {b}")
        cells_b: dict = {}
        qarrs: dict = {}
        for j in steer:
            cid = j["cell_id"]
            cq = _q_arr(j)
            qarrs[cid] = (j, cq)
            point, lo, hi = _boot_diff_ci(cq, a0_q, _boot_idx(len(a0_q), N_BOOT_CELL, cid + "__w2"))
            cells_b[cid] = {
                "cell": j["cell"],
                "delta_score": point,
                "ci_frozen": [lo, hi],
                "ci_label": f"frozen (decisive grid, n_q={len(a0_q)})",
                "coherence_pass": j["coherence_pass"],
                "frac_items_complete": j["accounting"]["frac_items_complete"],
            }
        null_ctx = [
            cq
            for (j, cq) in qarrs.values()
            if j["cell"]["position"] == "context"
            and j["cell"]["direction"] in NULL_STEER["context"]
            and j["coherence_pass"]
        ]
        band = _null_band(null_ctx, a0_q, f"{b}__w2nullctx", n_draws=N_BOOT_VERDICT)

        def _best(d, p, qarrs=qarrs, cells_b=cells_b):
            cands = [
                (cells_b[cid]["delta_score"], cid, cq)
                for cid, (j, cq) in qarrs.items()
                if j["cell"]["direction"] == d
                and j["cell"]["position"] == p
                and j["coherence_pass"]
            ]
            return max(cands, key=lambda kv: kv[0]) if cands else None

        pre = _best("pre", "context")
        ctxext = _best("ctxext", "context")
        margins: dict = {}
        if band is not None:
            bp = band["p975"]
            if pre is not None:
                ci = cells_b[pre[1]]["ci_frozen"]
                margins["E_pre"] = {
                    "value": pre[0] - bp,
                    "cell_id": pre[1],
                    "ci": [ci[0] - bp, ci[1] - bp],
                    "band_p975": bp,
                }
            if ctxext is not None:
                ci = cells_b[ctxext[1]]["ci_frozen"]
                margins["E_ctxdir"] = {
                    "value": ctxext[0] - bp,
                    "cell_id": ctxext[1],
                    "ci": [ci[0] - bp, ci[1] - bp],
                    "band_p975": bp,
                }
        if pre is not None and ctxext is not None:
            point, lo, hi = _boot_diff_ci(
                pre[2], ctxext[2], _boot_idx(len(a0_q), N_BOOT_VERDICT, f"{b}__cgap")
            )
            margins["C_gap"] = {"value": point, "ci": [lo, hi]}
        label, reason = _lattice_label(margins)
        sel = {}
        for d, p in (("pre", "context"), ("ctxext", "context"), ("rb", "answer")):
            si = _selection_inherited_for(out_root, b, d, p)
            if si is not None:
                sel[f"{d}__{p}"] = si
        verdicts["behaviors"][b] = {
            "label": label,
            "reason": reason,
            "margins": margins,
            "null_band_context": band,
            "selection_inherited": sel,
            "ci_label_note": (
                f"margin CIs are frozen-at-operating-point (decisive grid, n_q={len(a0_q)}) "
                "shifted by the null band; selection_inherited entries re-argmax the "
                "localize grid per draw"
            ),
        }
        percell_out["behaviors"][b] = cells_b
    _write_json_atomic(
        out_root / "decisive" / "delta_score_percell.json", _run_metadata(percell_out)
    )
    _write_json_atomic(out_root / "decisive" / "verdicts.json", _run_metadata(verdicts))
    _write_json_atomic(
        out_root / "patch" / "patch_vs_ceiling.json", _run_metadata(_patch_vs_ceiling(out_root))
    )
    files = sorted((out_root / "judge" / "decisive" / "judged").glob("*.json")) + sorted(
        (out_root / "judge" / "patch" / "judged").glob("*.json")
    )
    _write_json_atomic(
        out_root / "judge" / "completeness_wave2.json", _run_metadata(_completeness_block(files))
    )
    _upload_folder_to_hf(
        out_root / "decisive",
        f"{_hf_prefix()}/decisive",
        allow=["delta_score_percell.json", "verdicts.json", "selection_meta.json"],
    )
    _upload_folder_to_hf(
        out_root / "patch",
        f"{_hf_prefix()}/patch",
        allow=["patch_vs_ceiling.json", "calibration_projections.json"],
    )


def _upload_judge_outputs(out_root: Path, phases) -> None:
    """Pack EVERY per-cell judge tree (judged/cache/raw) into <=9 MB plain
    JSONL line-shards (rw2220 packer; never gzip — *.gz is LFS-matched) and
    upload ONLY the packed shard dirs. The shared data repo sits at the
    Hub's 1,000,000-file REPO ceiling (#2286), so a per-cell upload of
    O(1000) files is rejected by the commit endpoint outright — wave 1 died
    exactly there (localize judged/ = 1,155 files). Pilot reports + pass
    sidecars upload alongside (few files by allow-pattern)."""
    _ensure_repo_root_on_syspath()
    import scripts.issue2220_readwrite as rw2220

    for phase in phases:
        base = out_root / "judge" / phase
        for sub in ("judged", "cache", "raw"):
            src = base / sub
            if not src.exists():
                continue
            dest = base / f"{sub}_pack"
            # save_raw writes ONE bare-<cid> EXTENSIONLESS JSON file per cell
            # (judging.judge_items_graded save_raw path) — the packer's default
            # *.json rglob packs ZERO rows there (silent raw-drop, caught in
            # the wave-1 recovery: raw_pack shipped n_files=0 manifests).
            pattern = "*" if sub == "raw" else "*.json"
            rw2220._pack_tree_to_jsonl_shards(src, dest, group=f"{phase}_{sub}", pattern=pattern)
            _upload_folder_to_hf(
                dest, f"{_hf_prefix()}/judge/{phase}/{sub}_pack", allow=["*.jsonl", "*.json"]
            )
    pilot = out_root / "judge" / "pilot"
    if pilot.exists():
        _upload_folder_to_hf(
            pilot, f"{_hf_prefix()}/judge/pilot", allow=["*.report.json", "*.pass.json"]
        )


def phase_judge_reduce(args) -> None:
    """Off-pod judge + reduce (plan §4.2): --reduce-phase localize = wave 1
    (judge baseline_ceiling + localize -> dose_response / operating_points /
    gates); --reduce-phase decisive = wave 2 (judge decisive + patch ->
    percell / verdicts / patch_vs_ceiling). Batch-API judging via
    judge_items_graded; grid-completeness assert BEFORE any judge spend
    (`_assert_gen_grid_complete` — never a partial-grid reduce); rule-26
    pilot per behavior BEFORE the bulk wave; per-cell judged checkpoints;
    VM/CPU-only (no CUDA). Stated deviation: coherence gating is programmatic
    (steering.coherence_check) — the judged coherence covariate is not
    collected (recorded in gates.json, mirrors #2220)."""
    from explore_persona_space.experiments.issue_1739.judging import load_trait_rubric

    out_root = _out_root(args)
    wave = args.reduce_phase
    phases = _WAVE_SRC[wave]
    behaviors = list(args.behaviors)
    rubrics = {b: load_trait_rubric(b) for b in behaviors}
    comp_roots = {phase: _stage_phase_completions(out_root, phase) for phase in phases}
    pilot_behaviors = _assert_gen_grid_complete(args, out_root, wave, comp_roots)
    steered = "localize" if wave == "localize" else "decisive"
    for b in pilot_behaviors:
        _run_judge_pilot(args, out_root, steered, b, rubrics[b], _judge_draws(args, steered))
    for phase in phases:
        files = sorted(comp_roots[phase].glob("*.json"))
        if not files:
            raise RuntimeError(f"judge_reduce: no gen cells for {phase}")
        n_draws = _judge_draws(args, phase)
        t0 = time.time()
        for k, f in enumerate(files, 1):
            b = f.name.split("__", 1)[0]
            if b not in rubrics:
                continue
            _judge_cell(args, out_root, phase, f, rubrics[b], n_draws)
            _progress(f"judge_{phase}", k, len(files), f.stem, t0)
    if wave == "localize":
        _reduce_wave1(args, out_root)
    else:
        _reduce_wave2(args, out_root)
    _upload_judge_outputs(out_root, phases)
    _write_sentinel(out_root, f"judge_reduce_{wave}", "done", {"phases": list(phases)})
    _breadcrumb("judge_reduce", wave=wave, status="done")


# ---------------------------------------------------------------------------
# phase: figures (off-pod VM; scripts/issue2254_figures.py)
# ---------------------------------------------------------------------------


def phase_figures(args) -> None:
    """Render every figure whose reduce inputs exist (scripts/issue2254_figures
    .render_all); skipped figures are named with reasons, never silent."""
    _ensure_repo_root_on_syspath()
    from scripts.issue2254_figures import render_all

    out_root = _out_root(args)
    res = render_all(out_root, Path(args.fig_dir))
    for name, reason in sorted(res["skipped"].items()):
        logger.info("[figures] skipped %s: %s", name, reason)
    if not res["rendered"]:
        raise RuntimeError("figures: zero figures rendered — no reduce outputs found")
    _breadcrumb("figures", rendered=len(res["rendered"]), skipped=len(res["skipped"]))
    _write_sentinel(
        out_root,
        "figures",
        "done",
        {"rendered": sorted(res["rendered"]), "skipped": sorted(res["skipped"])},
    )


UNIT3_PHASES = (
    "baseline_ceiling",
    "localize",
    "decisive",
    "patch",
    "build_pools",
    "margin",
    "judge_reduce",
    "figures",
)

PHASES = {
    "stage_inputs": phase_stage_inputs,
    "fit_maps": phase_fit_maps,
    "capture_directions": phase_capture_directions,
    "norm_probe": phase_norm_probe,
    "baseline_ceiling": phase_baseline_ceiling,
    "localize": phase_localize,
    "decisive": phase_decisive,
    "patch": phase_patch,
    "build_pools": phase_build_pools,
    "margin": phase_margin,
    "judge_reduce": phase_judge_reduce,
    "figures": phase_figures,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="issue #2254 pre-image steering driver")
    ap.add_argument("--phase", choices=sorted(PHASES), help="phase to run")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--layers", nargs="+", type=int, default=list(ALL_LAYERS))
    ap.add_argument("--out-root", default="eval_results/issue_2254")
    ap.add_argument(
        "--fit-workers",
        type=int,
        default=max(1, min(14, os.cpu_count() or 1)),
        help="ProcessPool width for the per-layer map fits (plan §9: 16 vCPU pod)",
    )
    ap.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="round-robin cell shard for the multi-GPU fan-out (unit-3 phases; plan §9)",
    )
    ap.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="total shards; launcher pins CUDA_VISIBLE_DEVICES per shard",
    )
    ap.add_argument(
        "--pilot-questions",
        type=int,
        default=PILOT_N_QUESTIONS,
        help="timing-pilot question count (production localize grain = 10)",
    )
    ap.add_argument(
        "--pilot-draws",
        type=int,
        default=PILOT_N_DRAWS,
        help="timing-pilot draws per question (production localize grain = 3)",
    )
    ap.add_argument(
        "--reduce-phase",
        choices=("localize", "decisive"),
        default="localize",
        help="judge_reduce wave: localize = wave 1 (gates), decisive = wave 2 (verdicts)",
    )
    ap.add_argument(
        "--q-localize",
        type=int,
        default=10,
        help="eval questions per localize cell (plan §4.2: 10)",
    )
    ap.add_argument(
        "--q-decisive",
        type=int,
        default=N_EVAL_QUESTIONS,
        help="eval questions per decisive/baseline/patch cell (plan §4.2: 20)",
    )
    ap.add_argument(
        "--draws-localize", type=int, default=3, help="draws per question, localize grid"
    )
    ap.add_argument(
        "--draws-decisive",
        type=int,
        default=5,
        help="draws per question, decisive/baseline/patch-projection cells",
    )
    ap.add_argument("--fig-dir", default="figures/issue_2254", help="figures output dir")
    ap.add_argument("--force", action="store_true", help="ignore per-phase caches (unit-3 phases)")
    ap.add_argument(
        "--smoke", action="store_true", help="tiny slice (1 behavior, layer 14, 2q x 2 draws)"
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="enumerate the phase grid + resolve deferred imports, no GPU/HF/model",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="AST arg-attribute completeness check, then exit 0",
    )
    return ap


def _apply_smoke(args) -> None:
    """Tiny-real slice (plan §4.4 smoke (i)): 1 behavior, layer 14 (the evil
    frozen parity layer, so the fit_maps smoke exercises the HALT parity gate
    end-to-end), pilot 2 questions x 2 draws. Scratch out-root + smoke/ HF
    sub-prefix so smoke artifacts never overwrite canonical ones."""
    global _SMOKE_UPLOAD_SUBPREFIX
    args.behaviors = args.behaviors[:1]
    args.layers = [PILOT_LAYER]
    args.pilot_questions = 2
    args.pilot_draws = 2
    args.q_localize = 2
    args.q_decisive = 2
    args.draws_localize = 2
    args.draws_decisive = 2
    if args.out_root == "eval_results/issue_2254":
        args.out_root = "/tmp/issue-2254-smoke"
    if args.fig_dir == "figures/issue_2254":
        args.fig_dir = "/tmp/issue-2254-smoke/figures"
    _SMOKE_UPLOAD_SUBPREFIX = True


def _dry_run_phase(args) -> None:
    """Enumerate the phase's grid + RESOLVE its deferred imports (no GPU/HF/
    model): a missing symbol / signature drift in a pod-only branch must fail
    HERE, not after the expensive phases (#606/#823)."""
    phase = args.phase
    if phase == "stage_inputs":
        from huggingface_hub import hf_hub_download  # noqa: F401

        _ensure_repo_root_on_syspath()
        import scripts.issue2220_readwrite as rw2220
        from scripts.issue952_stats import _reconstruct_lmsys_prompts

        assert callable(rw2220._assert_eval_bank_disjoint)
        assert callable(rw2220._norm_question)
        assert callable(rw2220._eval_questions)
        assert callable(_reconstruct_lmsys_prompts)
        _breadcrumb("stage_inputs", dry_run=1, behaviors=len(args.behaviors))
    elif phase == "fit_maps":
        from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
            identity_bias_predict,
            knn_retrieval,
        )
        from explore_persona_space.experiments.issue_1739 import store_io
        from explore_persona_space.orchestrate.preflight import (  # noqa: F401
            assert_out_root_headroom,
        )

        assert callable(store_io.load_rb_bank)
        _breadcrumb(
            "fit_maps",
            dry_run=1,
            fits=2 * len(args.layers),
            layers=len(args.layers),
        )
    elif phase == "capture_directions":
        from explore_persona_space.experiments.issue1415 import steering

        assert callable(steering.capture_vectors)
        n_ctx = 2 * N_INSTRUCTION_PAIRS * N_EXTRACTION_QUESTIONS
        _breadcrumb(
            "capture_directions",
            dry_run=1,
            contexts_per_behavior=n_ctx,
            dirs=len(args.behaviors) * 6 * len(args.layers),
        )
    elif phase == "norm_probe":
        from explore_persona_space.analysis.extraction import (  # noqa: F401
            extract_layer_activations,
        )
        from explore_persona_space.experiments.issue1415 import steering

        assert callable(steering.generate_batch)
        assert callable(steering.context_token_ids)
        _breadcrumb(
            "norm_probe",
            dry_run=1,
            probes=len(args.behaviors) * len(args.layers),
            pilot_completions=args.pilot_questions * args.pilot_draws,
        )
    elif phase in ("baseline_ceiling", "localize", "decisive", "patch"):
        import inspect

        from explore_persona_space.experiments.issue1415 import steering
        from explore_persona_space.experiments.issue2254.hooks import (
            ProjectionPatchHook,
            multi_layer_delta_hooks,
            multi_layer_projection_patch_hooks,
        )

        assert callable(multi_layer_delta_hooks)
        assert callable(ProjectionPatchHook)
        assert callable(multi_layer_projection_patch_hooks)
        # Signature-bind the smoke-fenced generate_batch call shape (#606/#1332).
        inspect.signature(steering.generate_batch).bind(
            None, None, ["c"], n=1, hook=None, max_new_tokens=8, seed_base=42
        )
        if phase == "localize":
            cells = _localize_cells(args, list(args.behaviors))
            _breadcrumb(phase, dry_run=1, cells=len(cells))
        elif phase == "baseline_ceiling":
            _breadcrumb(phase, dry_run=1, cells=2 * len(args.behaviors))
        elif phase == "patch":
            n = len(PATCH_DIRECTIONS) * len(PATCH_OPS) * len(PATCH_BREADTHS)
            assert callable(steering.capture_vectors)
            _breadcrumb(phase, dry_run=1, cells_per_behavior=n)
        else:  # decisive: cells need localize/operating_points.json — count combos only
            n = len(_grid_combos(args)) * len(BREADTHS) + 1
            _breadcrumb(phase, dry_run=1, max_cells_per_behavior=n)
    elif phase == "build_pools":
        from explore_persona_space.orchestrate import hub

        assert callable(hub.stage_hub_file)
        _breadcrumb(phase, dry_run=1, behaviors=len(args.behaviors))
    elif phase == "margin":
        import inspect

        _ensure_repo_root_on_syspath()
        import scripts.issue2220_readwrite as rw2220

        # Bind the exact production call shape against the reused helper.
        inspect.signature(rw2220._batched_ln_logp).bind(
            None, [1], [[1]], None, 14, 0.0, "context", pad_id=0, batch_size=MARGIN_BATCH_2254
        )
        assert callable(rw2220._pack_tree_to_jsonl_shards)
        _breadcrumb(phase, dry_run=1, behaviors=len(args.behaviors))
    elif phase == "judge_reduce":
        import inspect

        from explore_persona_space.eval.judge_pilot import judge_pilot_gate
        from explore_persona_space.experiments.issue_1739.judging import (
            judge_items_graded,
            judge_tallies,
            load_trait_rubric,
            rollout_item_id,
        )

        assert callable(judge_tallies) and callable(load_trait_rubric)
        assert rollout_item_id("a-b", 0) == "a-b_k00"
        inspect.signature(judge_pilot_gate).bind(
            {"a": []},
            "rubric",
            max_tokens=JUDGE_MAX_TOKENS_2254,
            cache_dir=Path("."),
            save_raw_dir=Path("."),
            n_draws=2,
            target_total_draws=10,
            min_effective_draws_per_arm=JUDGE_PILOT_MIN_EFFECTIVE,
            allow_subresolution_pilot=True,
            report_path=None,
        )
        inspect.signature(judge_items_graded).bind(
            [],
            "rubric",
            cache_dir=Path("."),
            save_raw=Path("."),
            n_draws=2,
            temperature=1.0,
            max_tokens=JUDGE_MAX_TOKENS_2254,
            judge_model="m",
        )
        _breadcrumb(phase, dry_run=1, wave=args.reduce_phase)
    elif phase == "figures":
        _ensure_repo_root_on_syspath()
        from scripts.issue2254_figures import render_all

        assert callable(render_all)
        _breadcrumb(phase, dry_run=1, fig_dir=args.fig_dir)
    else:
        raise SystemExit(f"dry-run: no wiring branch for phase {phase!r}")
    print(f"[dry-run] {phase} wiring OK", flush=True)


def main() -> None:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if args.phase is None:
        raise SystemExit("--phase is required (or --import-check)")
    if args.smoke:
        _apply_smoke(args)
    if args.dry_run:
        _dry_run_phase(args)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    PHASES[args.phase](args)
    # Explicit hard-exit after flush: this driver imports torch/transformers/HF,
    # so a finalize-time teardown race can rewrite the rc (gotchas.md). Outputs
    # are rename-atomic (_write_json_atomic: write_text + os.replace — no
    # fsync, so crash-safe vs partial writes, not power-loss durable) and
    # uploaded before here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
