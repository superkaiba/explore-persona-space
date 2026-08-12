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
  baseline_ceiling / localize / decisive / patch / margin / judge_reduce
                       unit-3 scope (pre-split contract) — NotImplementedError.

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
RHO_2220_JSON_REL = "eval_results/issue_2220/norm_probe/rho_by_layer.json"
RHO_2220_CONE = "eval_results/issue_2220/norm_probe"
RHO_PARITY_RTOL = 5e-3

# #2220 read-direction bank (Result 0 cosines; plan §10 reuse row).
RW2220_DIR_PREFIX = "issue2220_readwrite/directions"
RW2220_READ_SLUGS = ("mapread_ctx", "mapread_prefix")

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
    """Pod-observed sentinel (/workspace/logs/issue-2254-<phase>.json) the VM
    poller drains. Pod-side code NEVER shells to task.py."""
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

    Realized-keys assert (artifact-reuse check (c), plan §10): keys must
    include {cx_last, v_x, prompts, layers}; cx_last/v_x are (N, 28, 3584)
    fp16 with zero NaN/Inf; layers == list(range(28)); len(prompts) == N.
    Returns {"cx_last": tensor, "v_x": tensor, "prompts": list, "n_rows": N}.
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
    # weights_only=False: sha-pinned self-produced bundle carrying python lists
    # (prompts) alongside tensors — the torch>=2.6 weights_only default cannot
    # load it, and the revision pin is the trust boundary.
    tb = torch.load(path, map_location="cpu", weights_only=False)
    missing = {"cx_last", "v_x", "prompts", "layers"} - set(tb.keys())
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
    prompts = list(tb["prompts"])
    if len(prompts) != n:
        raise RuntimeError(f"pass-B prompts len {len(prompts)} != N rows {n}")
    logger.info("[pass-b] realized N=%d rows, 28 layers, H=%d (rev %s)", n, HIDDEN_DIM, HF_REV[:12])
    _BUNDLE_CACHE = {"cx_last": cx, "v_x": vx, "prompts": prompts, "n_rows": n}
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
    # the pass-B bundle's LMSYS prompts (normalized, set-compared, never
    # logged — content hygiene).
    import scripts.issue2220_readwrite as rw2220

    bundle = _load_pass_b_bundle()
    corpus_texts = {rw2220._norm_question(p) for p in bundle["prompts"] if p}
    if not corpus_texts:
        raise RuntimeError("A8: pass-B bundle yielded 0 prompt texts")
    record = rw2220._assert_eval_bank_disjoint(behaviors, corpus_texts=corpus_texts)
    record["corpus_source"] = f"pass-B LMSYS prompts ({PASS_B_FILE} @ {HF_REV[:12]})"
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
            "d_pre_raw_norm": float(np.linalg.norm(d_raw)),
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


def _upload_folder_to_hf(local_dir: Path, path_in_repo: str) -> None:
    """One bulk upload_folder commit (never a per-file loop); fail-loud, retried
    (issue2220_readwrite `_upload_directions` pattern)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    allow = ["*.pt", "*.json", "*.npz"]
    files = [
        p for p in local_dir.rglob("*") if p.is_file() and p.suffix in (".pt", ".json", ".npz")
    ]
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
        )
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
        geo_b = geometry[b]
        # ctxext cosine needs the captured direction; recover from saved bank
        ctx_path = dir_out / f"{b}_ctxext_L{ly}.pt"
        ctx_vec = torch.load(ctx_path, map_location="cpu", weights_only=False)["direction"]
        row["cos_ctxext"] = _cos(np.asarray(ctx_vec, dtype=np.float64), vec)
        assert geo_b is not None
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


def _rho_parity_assert(result: dict[str, dict[str, float]]) -> dict:
    """Rig-parity probe vs #2220's committed rho values at shared layers
    (plan §4.2). Matched single-row forward geometry => tight relative
    tolerance; a real failure (wrong position/layer/bank) is a >10% effect."""
    ref = json.loads(_ensure_git_input(RHO_2220_JSON_REL, RHO_2220_CONE).read_text())
    ref_rho = ref["rho_median_last_context_token"]
    checked, problems = [], []
    for behavior, ref_layers in ref_rho.items():
        ours = result.get(behavior)
        if ours is None:
            continue
        for lkey, ref_val in ref_layers.items():
            if lkey not in ours:
                continue
            rel = abs(ours[lkey] - float(ref_val)) / max(abs(float(ref_val)), 1e-12)
            checked.append(
                {
                    "behavior": behavior,
                    "layer": lkey,
                    "ours": ours[lkey],
                    "rw2220": ref_val,
                    "rel_dev": rel,
                }
            )
            if rel > RHO_PARITY_RTOL:
                problems.append(
                    f"{behavior}@{lkey}: {ours[lkey]:.4f} vs {ref_val:.4f} rel={rel:.2e}"
                )
    if not checked:
        raise RuntimeError(
            "HALT: rho parity probe found NO shared (behavior, layer) cells vs "
            f"{RHO_2220_JSON_REL} — rig parity cannot be skipped (plan §4.2)"
        )
    if problems:
        raise RuntimeError(
            f"HALT: rho parity vs #2220 FAILED (rtol {RHO_PARITY_RTOL}): " + "; ".join(problems)
        )
    logger.info(
        "[norm_probe] rho parity PASS: %d shared cells within %.1e", len(checked), RHO_PARITY_RTOL
    )
    return {"n_checked": len(checked), "rtol": RHO_PARITY_RTOL, "cells": checked}


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


def phase_norm_probe(args) -> None:
    """rho_l = median last-context-token residual norm, all 28 layers, per
    behavior + pooled, with the #2220 shared-layer parity assert; then the
    1-cell timing pilot (plan §4.2 + §7 gate 1).

    Geometry note: per-context SINGLE-ROW unpadded forwards (the exact #2220
    phase_norm_probe geometry) so the parity assert compares like with like —
    a padded batched forward would shift bf16 numerics (gotchas.md matched
    batch geometry).
    """
    _require_cuda("norm_probe")
    import torch

    from explore_persona_space.analysis.extraction import extract_layer_activations
    from explore_persona_space.experiments.issue1415 import steering

    out_root = _out_root(args)
    _assert_phase_headroom(out_root, 1.0, "norm_probe")
    layers = sorted(int(x) for x in args.layers)
    behaviors = list(args.behaviors)
    _breadcrumb("norm_probe", behaviors=len(behaviors), layers=len(layers))
    model, tok = _load_model_and_tokenizer()

    result: dict[str, dict[str, float]] = {}
    pooled: dict[int, list[float]] = {ly: [] for ly in layers}
    t0 = time.time()
    for bi, behavior in enumerate(behaviors, 1):
        questions = _eval_questions(behavior)
        contexts = _contexts_for_questions(questions)
        norms = {ly: [] for ly in layers}
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
        _progress("norm_probe", bi, len(behaviors), behavior, t0)

    parity = _rho_parity_assert(result)
    payload = _run_metadata(
        {
            "rho_median_last_context_token": result,
            "rho_pooled_median": {f"L{ly}": float(np.median(pooled[ly])) for ly in layers},
            "layers": layers,
            "rw2220_parity": parity,
        }
    )
    _write_json_atomic(out_root / "norm_probe" / "rho_by_layer.json", payload)

    pilot = _timing_pilot(args, model, tok, result)
    _write_json_atomic(out_root / "norm_probe" / "timing_pilot.json", _run_metadata(pilot))

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
# unit-3 phases (pre-split contract): explicit NotImplementedError stubs
# ---------------------------------------------------------------------------

UNIT3_PHASES = ("baseline_ceiling", "localize", "decisive", "patch", "margin", "judge_reduce")


def _unit3_stub(name: str):
    def _stub(args) -> None:
        raise NotImplementedError(
            f"--phase {name}: unit-3 scope (pre-split implementation contract; "
            "plan §4 phase order). Units 1-2 shipped hooks + stage_inputs/"
            "fit_maps/capture_directions/norm_probe; unit 3 fills this phase."
        )

    _stub.__name__ = f"phase_{name}_stub"
    return _stub


PHASES = {
    "stage_inputs": phase_stage_inputs,
    "fit_maps": phase_fit_maps,
    "capture_directions": phase_capture_directions,
    "norm_probe": phase_norm_probe,
    **{name: _unit3_stub(name) for name in UNIT3_PHASES},
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
    if args.out_root == "eval_results/issue_2254":
        args.out_root = "/tmp/issue-2254-smoke"
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

        assert callable(rw2220._assert_eval_bank_disjoint)
        assert callable(rw2220._norm_question)
        assert callable(rw2220._eval_questions)
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
    else:
        _breadcrumb(phase, dry_run=1, unit3_stub=1)
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
    # are fsynced (_write_json_atomic) + uploaded before here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
