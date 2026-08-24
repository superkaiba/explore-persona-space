"""Issue #2254 same-issue follow-up round 6 `transpose_ladder` — thin driver.

Plan v14 (tasks/.../2254/plans/plan.md). Single manipulated variable
(plan §4.1): the DIRECTION CONSTRUCTION — four forward/ridge-weighted map
pullbacks of r_B under the parent's stored per-layer context→answer maps,
replacing nothing (the parent's k*-truncated pinv rows stand as the external
inverse-weighted reference):

- ``tr``  (transpose):      w = Vmtᵀ (Sm ⊙ c),                 c = Umᵀ r_B
- ``rl1``/``rl2``/``rl3``:  w = Vmtᵀ ((Sm/(Sm²+λ_j)) ⊙ c),     λ_j = q{05,50,95}({Sm_i²})

full spectrum, per layer, behavior-independent λ; destandardization mirrors
the parent EXACTLY (``i2254.destandardized_direction``). Everything else —
grid (44 cells at the parent's context-locus operating points), generation
(20q × 5 draws × seeds {42,43}, cap 2048, >2% cap-hit regen at 2×), judge
instrument (Sonnet 4.5, graded 0-100, thr 50, 5 draws, Batch API + rule-26
pilot + rule-28 sync re-issue), reduce (question-clustered paired bootstrap
vs the parent floor, margins vs the parent band) — is INHERITED verbatim.

Phases: ``directions`` (CPU: 28 float64 SVDs of M=Wᵀ, 224 direction .pt files,
HALT-class parity gates (i)/(ii)/(iii), ladder_report.json) → ``steer`` (GPU:
44 cells round-robin sharded, per-cell JSON checkpoints, cached-skip resume,
packed HF raw-completion upload BEFORE the shard sentinel) → ``judge``
(off-pod Batch API via ``judge_items_graded``; pilot gate; rule-28 re-issue)
→ ``reduce`` (VM CPU: Δ vs floor, margin vs band, §3 verdict lattice,
two-grain multiplicity tags, selection-aware companions, intrusion
sensitivity, ``fresh_nulls: false``) → ``figures``
(``scripts.issue2254_ladder_figures``). Plus ``--cpu-smoke`` (VM, no GPU/API)
and ``--rig-health`` (plan §4 smoke item (d), advisory only).

Reuse map (import, never copy-paste bodies — the v10 first-k precedent):
``map_svd`` / ``preimage_w`` / ``destandardized_direction`` /
``kstar_from_fit`` / ``_save_direction`` / ``_ensure_direction_vec`` /
``_steer_hook_factory`` / ``_gen_cell_rows`` / ``_load_rho`` /
``_load_operating_points`` / ``_eval_questions`` / ``_contexts_for_questions``
/ ``_run_judge_pilot`` / ``_judge_ctx_id`` / ``_boot_idx`` / ``_boot_diff_ci``
/ sentinel + checkpoint + upload machinery from ``scripts/issue2254_preimage``;
``_judge_graded_with_refusal_reissue`` / ``_judge_instrument_fp`` /
``_assert_hub_headroom_for_steer`` / ``_wipe_stale_sentinels`` / ``_hub_tree``
/ ``_hub_stage`` from ``scripts/issue2254_first_k_steering``;
``_pack_tree_to_jsonl_shards`` from ``scripts/issue2220_readwrite``.

Conventions: fail fast (no silent defaults, no except-pass); content hygiene —
question/completion text lands in JSON payloads only, never in logs; reused
INPUTS resolve at canonical committed locations in BOTH modes; only OUTPUT
roots + the HF sub-prefix rebind under --smoke. The ``--cpu-smoke`` harness
rebinds the module seams ``_RB_LOADER`` / ``_PARENT_VEC_LOADER`` / ``_UPLOAD``
/ ``_TOKENIZER_LOADER`` to fixture-backed equivalents (disclosed here; the
POD smoke — plan §4 items (a)-(d) — runs the unmodified production seams).
"""

from __future__ import annotations

import os

# HF transfer accelerators BEFORE any huggingface_hub import (upload-policy).
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

# load_dotenv BEFORE any numpy/torch import (thread-cap + credential
# setdefaults freeze at BLAS/torch import; orchestrate.env, never bare dotenv).
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()


def _ensure_repo_root_on_syspath() -> None:
    """Repo root on sys.path so `import scripts.<mod>` resolves (#823)."""
    repo_root = Path(__file__).resolve().parents[1]
    assert (repo_root / "pyproject.toml").exists(), f"repo-root sentinel missing at {repo_root}"
    p = str(repo_root)
    if p not in sys.path:
        sys.path.insert(0, p)


_ensure_repo_root_on_syspath()

import numpy as np  # noqa: E402  (after load_dotenv so BLAS thread caps apply)

import scripts.issue2254_first_k_steering as fk  # noqa: E402  (judge/pack reuse)
import scripts.issue2254_preimage as i2254  # noqa: E402

logger = logging.getLogger("issue2254.ladder")

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# round constants (plan v14 §4 / §6 / §9 / §10)
# ---------------------------------------------------------------------------

FOLLOWUP_LABEL = "transpose_ladder"
ROUND_BEHAVIORS = ("evil", "sycophancy")  # hallucination stays gate-2 demoted (§2)
LADDER_SLUGS = ("tr", "rl1", "rl2", "rl3")
LAMBDA_QUANTILES = {"rl1": 0.05, "rl2": 0.50, "rl3": 0.95}  # of {Sm_i²}, per layer (§4.1)
UNION_SOURCE_ARMS = ("pre", "rb", "ctxext")  # ops-union sources (§4.2)

# Pre-registered cell family (§4.2, quoted from operating_points.json at plan
# time). `registered_grid_from_ops` re-derives this from the artifact and
# ASSERTS equality — drift in either direction fails loud.
PLAN_GRID: dict[str, tuple[tuple[str, float], ...]] = {
    "evil": (("L14", 0.5), ("L14", 4.0), ("mid", 0.5), ("mid", 2.0), ("mid", 4.0), ("all", 0.5)),
    "sycophancy": (("L17", 4.0), ("L14", 2.0), ("mid", 2.0), ("mid", 4.0), ("all", 4.0)),
}
FAMILY_SIZE = 44  # 11 grid points x 4 arms (§3 registered family)
ARM_SIZE = 11  # cells per arm across both behaviors
ALPHA = 0.05  # §6 multiplicity-tag base level

PARITY_LAYERS = (14, 17)  # gate (ii) cells: 2 behaviors x {14, 17} (§4.1)
PARITY_COS_MIN = 0.999  # gate (ii) threshold (§11)
RESIDUAL_RTOL = 1e-6  # gate (iii) relative residual ceiling (§4.1)
LOADER_ROUNDTRIP_MIN_COS = 1.0 - 1e-9  # gate (iii) production-loader round-trip
AMP_TRIPWIRE_RATIO = 1e6  # §12.16 degenerate-amplification tripwire on |w|
CONC_KS = (10, 100)  # alignment-concentration ks (plus k*)

NPZ_KEYS = {"W", "kstar", "lam", "n_rows", "pass_b_revision", "s", "xmu", "xsd", "ymu"}

Q_STEER_DEFAULT = i2254.N_EVAL_QUESTIONS  # 20 (§4.2)
DRAWS_DEFAULT = i2254.JUDGE_DRAWS["decisive"]  # 5 gen draws/question/seed (§4.2 parent grain)
LADDER_SEEDS = i2254.SEEDS_DECISIVE  # (42, 43) — parent decisive parity (§4.2)

SENTINEL_DIRECTIONS = "ladder-directions"
SENTINEL_STEER = "ladder-steer"
SENTINEL_FIGURES = "ladder-figures"

PACK_FLUSH_EVERY = 8  # cells between incremental pack+upload flushes
LADDER_BYTES_PER_CELL = 2_500_000  # ~200 completions/cell at the 2048 cap (fk sizing basis)

# Reused parent inputs resolve at CANONICAL, --out-root-INDEPENDENT locations
# (smoke-root-rebinding gotcha: inputs are never smoke-diverted).
INPUTS_ROOT = _REPO_ROOT / "eval_results" / "issue_2254"
GIT_INPUTS = (
    ("eval_results/issue_2254/localize/operating_points.json", "eval_results/issue_2254/localize"),
    ("eval_results/issue_2254/norm_probe/rho_by_layer.json", "eval_results/issue_2254/norm_probe"),
    (
        "eval_results/issue_2254/baseline_ceiling/judged_percell.json",
        "eval_results/issue_2254/baseline_ceiling",
    ),
    ("eval_results/issue_2254/decisive/verdicts.json", "eval_results/issue_2254/decisive"),
    (
        "eval_results/issue_2254/decisive/delta_score_percell.json",
        "eval_results/issue_2254/decisive",
    ),
    ("eval_results/issue_2254/decisive/cjk_audit.json", "eval_results/issue_2254/decisive"),
)
COMMON_HORIZON_TOKENS = 2048  # intrusion recount horizon (r5 sensitivity convention)

# §12.19 parent-reference fixture: the published evil measured-context-direction
# margin the reduce must REPRODUCE from parent artifacts before any new cell.
PARENT_REFERENCE_CELL = "evil__cxd__ctx__L14__c4"
PARENT_REFERENCE_MARGIN = 2.458555555555556


class LadderHaltError(RuntimeError):
    """HALT-class gate failure (plan §4.1/§7): kill the round before GPU spend."""


def round_root(out_root: Path) -> Path:
    """This round's OUTPUT root under the issue out-root (rebinds under --smoke)."""
    return Path(out_root) / FOLLOWUP_LABEL


def _round_hf_prefix() -> str:
    """HF prefix for round OUTPUTS (smoke-diverted via the parent flag)."""
    return f"{i2254._hf_prefix()}/{FOLLOWUP_LABEL}"


def _ladder_metadata(extra: dict) -> dict:
    """Parent reproducibility envelope + the round label."""
    return i2254._run_metadata({"followup_label": FOLLOWUP_LABEL, **extra})


# ---------------------------------------------------------------------------
# module seams (production defaults; --cpu-smoke / tests rebind — disclosed in
# the module docstring; the POD smoke runs the unmodified defaults)
# ---------------------------------------------------------------------------


def _rb_loader_production() -> dict[str, np.ndarray]:
    """#779 r_B bank at the pin -> {behavior: (28, H) float64} (parent loader)."""
    return i2254._load_rb_all()


def _parent_vec_canonical(bank_root: Path, behavior: str, slug: str, layer: int):
    """Parent bank vector (pre/ctxext) — CANONICAL HF prefix by construction
    (i2254.HF_PREFIX, never the smoke sub-prefix: reused inputs are not
    smoke-diverted; the v10 DirectionBank convention). Local-first at
    ``bank_root/directions``; unit-norm fp32 (H,) torch tensor."""
    import torch

    from explore_persona_space.orchestrate import hub

    name = f"{behavior}_{slug}_L{layer}.pt"
    path = Path(bank_root) / "directions" / name
    if not path.exists():
        hub.stage_hub_file(
            i2254.HF_DATA_REPO,
            f"{i2254.HF_PREFIX}/directions/{name}",
            path,
            repo_type="dataset",
        )
    payload = torch.load(path, map_location="cpu", weights_only=True)
    vec = payload["direction"].float()
    assert vec.shape == (i2254.HIDDEN_DIM,), (name, tuple(vec.shape))
    return vec / vec.norm()


def _tokenizer_production():
    """Qwen tokenizer for the intrusion common-horizon recount (r5 convention)."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(i2254.MODEL_NAME)


_RB_LOADER = _rb_loader_production
_PARENT_VEC_LOADER = _parent_vec_canonical
_UPLOAD = i2254._upload_folder_to_hf
_TOKENIZER_LOADER = _tokenizer_production


# ---------------------------------------------------------------------------
# direction construction — pure algebra (CPU-tested;
# tests/test_issue2254_transpose_ladder.py)
# ---------------------------------------------------------------------------


def ladder_lambdas(Sm) -> dict[str, float]:
    """Registered λ rule (§4.1): λ_j = quantile_j({Sm_i²}) at j ∈ {.05,.50,.95},
    full spectrum, per layer, behavior-independent. Deterministic (np.quantile
    linear interpolation on the sorted squared map spectrum)."""
    s2 = np.asarray(Sm, dtype=np.float64) ** 2
    return {slug: float(np.quantile(s2, q)) for slug, q in LAMBDA_QUANTILES.items()}


def ladder_weight(Um, Sm, Vmt, r_b, slug: str, lam: float | None = None) -> np.ndarray:
    """Standardized-frame ladder weight (§4.1): per-mode c_i·s_i for ``tr``,
    s_i/(s_i²+λ) for the ridge rungs; c = Umᵀ r_B; full spectrum."""
    c = np.asarray(Um, dtype=np.float64).T @ np.asarray(r_b, dtype=np.float64)
    s = np.asarray(Sm, dtype=np.float64)
    if slug == "tr":
        return np.asarray(Vmt, dtype=np.float64).T @ (s * c)
    if slug not in LAMBDA_QUANTILES:
        raise ValueError(f"unknown ladder slug {slug!r}")
    if lam is None or not (np.isfinite(lam) and lam > 0.0):
        raise ValueError(f"ridge rung {slug} requires a positive finite lambda, got {lam!r}")
    return np.asarray(Vmt, dtype=np.float64).T @ ((s / (s**2 + float(lam))) * c)


def transpose_residual(M, w, r_b) -> float:
    """Gate (iii) transpose check: relative residual ‖w − Mᵀr_B‖/‖Mᵀr_B‖."""
    ref = np.asarray(M, dtype=np.float64).T @ np.asarray(r_b, dtype=np.float64)
    den = float(np.linalg.norm(ref))
    if den <= 0.0:
        raise LadderHaltError("transpose_residual: degenerate reference Mᵀr_B")
    return float(np.linalg.norm(np.asarray(w, dtype=np.float64) - ref) / den)


def ridge_residual(M, w, r_b, lam: float) -> float:
    """Gate (iii) ridge check: relative residual of (MᵀM + λI)w = Mᵀr_B, via
    two matvecs (never a dense MᵀM materialization — plan §4.1)."""
    M64 = np.asarray(M, dtype=np.float64)
    w64 = np.asarray(w, dtype=np.float64)
    ref = M64.T @ np.asarray(r_b, dtype=np.float64)
    den = float(np.linalg.norm(ref))
    if den <= 0.0:
        raise LadderHaltError("ridge_residual: degenerate reference Mᵀr_B")
    lhs = M64.T @ (M64 @ w64) + float(lam) * w64
    return float(np.linalg.norm(lhs - ref) / den)


def alignment_concentration(c, kstar: int) -> dict[str, float]:
    """‖c[:k]‖/‖c‖ at k ∈ {10, 100, k*} (§4.1 diagnostics; c = Umᵀ r_B)."""
    c64 = np.asarray(c, dtype=np.float64)
    tot = float(np.linalg.norm(c64))
    if tot <= 0.0:
        raise LadderHaltError("alignment_concentration: degenerate c = Umᵀ r_B")
    out: dict[str, float] = {}
    for k in (*CONC_KS, int(kstar)):
        kk = min(int(k), c64.shape[0])
        key = f"k{k}" if k in CONC_KS else "kstar"
        out[key] = float(np.linalg.norm(c64[:kk]) / tot)
    return out


def _cos(a, b) -> float:
    """Cosine between two vectors; fail-loud on degenerate norms."""
    a64 = np.asarray(a, dtype=np.float64)
    b64 = np.asarray(b, dtype=np.float64)
    den = float(np.linalg.norm(a64) * np.linalg.norm(b64))
    if den <= 0.0:
        raise LadderHaltError("_cos: degenerate operand norm")
    return float(a64 @ b64 / den)


# ---------------------------------------------------------------------------
# HALT-class gates (plan §4.1 (i)/(ii)/(iii); §7 kill criterion (a))
# ---------------------------------------------------------------------------


def assert_npz_keys(z, layer: int) -> None:
    """Phase-D re-assert of the plan-§10 realized npz key set on every layer."""
    got = set(z.files) if hasattr(z, "files") else set(z.keys())
    if got != NPZ_KEYS:
        raise LadderHaltError(
            f"L{layer:02d} npz keys {sorted(got)} != registered {sorted(NPZ_KEYS)} — "
            "stored-map schema drift (plan §10 realized-keys pin)"
        )


def halt_npz_selfconsistency(z, layer: int) -> None:
    """Gate (i): the stored fit spectrum must reproduce the stored k*
    (``kstar_from_fit(s, lam) == kstar``; the npz s/lam/kstar are READ-ONLY
    fit-side inputs — never refit)."""
    got = i2254.kstar_from_fit(np.asarray(z["s"], dtype=np.float64), float(z["lam"]))
    stored = int(z["kstar"])
    if got != stored:
        raise LadderHaltError(
            f"gate (i) FAIL at L{layer:02d}: kstar_from_fit(s, lam)={got} != stored "
            f"kstar={stored} — stored map is not self-consistent; HALT before any GPU spend"
        )


def rebuild_parent_preimage(z, rb_vec) -> np.ndarray:
    """Rebuild the parent's d_pre from the STORED W (fp32 round-trip) via the
    parent's own construction path (map_svd + preimage_w + destandardization)."""
    _M, Um, Sm, Vmt = i2254.map_svd(z["W"])
    w = i2254.preimage_w(Um, Sm, Vmt, np.asarray(rb_vec, dtype=np.float64), int(z["kstar"]))
    return i2254.destandardized_direction(z["xsd"], w)


def halt_rebuild_parity(d_rebuilt, bank_vec, behavior: str, layer: int) -> float:
    """Gate (ii): cos(d_pre rebuilt from stored W, committed bank d_pre) ≥
    0.999 — proves the fp32 W round-trip + our SVD path reproduce the parent's
    construction. Failure HALTs (no silent tolerance widening)."""
    cval = _cos(d_rebuilt, np.asarray(bank_vec, dtype=np.float64))
    if not (cval >= PARITY_COS_MIN):
        raise LadderHaltError(
            f"gate (ii) FAIL at {behavior}/L{layer}: pre-image rebuild cos={cval:.6f} < "
            f"{PARITY_COS_MIN} vs the committed bank file — construction pipeline does not "
            "reproduce the parent; HALT before any GPU spend"
        )
    return cval


def halt_amp_tripwire(w, behavior: str, slug: str, layer: int) -> None:
    """§12.16 degenerate-amplification tripwire: no |w| component may exceed
    1e6 × the median |w| (a raw-pinv-style blowup cannot pass silently)."""
    aw = np.abs(np.asarray(w, dtype=np.float64))
    med = float(np.median(aw))
    if med <= 0.0:
        raise LadderHaltError(f"{behavior}/{slug}/L{layer}: degenerate weight (median |w| == 0)")
    ratio = float(aw.max() / med)
    if ratio > AMP_TRIPWIRE_RATIO:
        raise LadderHaltError(
            f"{behavior}/{slug}/L{layer}: max|w|/median|w| = {ratio:.3g} > "
            f"{AMP_TRIPWIRE_RATIO:g} — degenerate amplification tripwire (plan §12.16)"
        )


# ---------------------------------------------------------------------------
# phase: directions (CPU; ProcessPool across layers)
# ---------------------------------------------------------------------------


def _maps_dir(args) -> Path:
    return Path(args.maps_dir) if args.maps_dir else Path(args.out_root) / "maps" / "perlayer"


def _stage_map_npz(layer: int, maps_dir: Path) -> Path:
    """Local-first per-layer map npz; else stage from the CANONICAL parent
    prefix (reused inputs are never smoke-diverted)."""
    from explore_persona_space.orchestrate import hub

    target = maps_dir / f"L{layer:02d}.npz"
    if not target.exists():
        hub.stage_hub_file(
            i2254.HF_DATA_REPO,
            f"{i2254.HF_PREFIX}/analysis_tensors/maps_perlayer/perlayer/L{layer:02d}.npz",
            target,
            repo_type="dataset",
        )
    return target


_WORKER_LIMITS = None  # keeps the threadpoolctl limiter alive per worker


def _worker_init(blas_threads: int) -> None:
    """ProcessPool initializer: cap BLAS threads per worker (plan §9 —
    8 workers × capped threads; an uncapped float64 SVD grabs every core)."""
    global _WORKER_LIMITS
    from threadpoolctl import threadpool_limits

    _WORKER_LIMITS = threadpool_limits(limits=max(1, int(blas_threads)))


def build_layer_ladder(task: dict) -> dict:
    """One layer's ladder constructions (ProcessPool worker; fail-fast).

    Runs gate (i), the SVD of M = Wᵀ, the registered λ rule (+ recompute
    equality assert), and per (behavior × slug): the weight, the gate-(iii)
    residual assert, the §12.16 tripwire, the destandardized fold, and the
    .pt save (parent ``_save_direction`` payload contract). Returns report
    rows + the folded float64 directions for main-process diagnostics."""
    layer = int(task["layer"])
    dir_out = Path(task["dir_out"])
    dir_out.mkdir(parents=True, exist_ok=True)
    z = np.load(task["npz_path"])
    assert_npz_keys(z, layer)
    halt_npz_selfconsistency(z, layer)
    xsd = np.asarray(z["xsd"], dtype=np.float64)
    M, Um, Sm, Vmt = i2254.map_svd(z["W"])
    lambdas = ladder_lambdas(Sm)
    for slug, q in LAMBDA_QUANTILES.items():
        registered = float(np.quantile(Sm.astype(np.float64) ** 2, q))
        if lambdas[slug] != registered:
            raise LadderHaltError(
                f"L{layer:02d}/{slug}: realized λ {lambdas[slug]!r} != registered "
                f"quantile {registered!r} (gate iii λ-rule pin)"
            )
    kstar = int(z["kstar"])
    manifest: list = []
    rows: dict[str, dict] = {}
    dirs: dict[str, np.ndarray] = {}
    for behavior, rb_vec in task["rb_rows"].items():
        rb64 = np.asarray(rb_vec, dtype=np.float64)
        c = Um.T @ rb64
        conc = alignment_concentration(c, kstar)
        for slug in LADDER_SLUGS:
            lam = lambdas.get(slug)
            w = ladder_weight(Um, Sm, Vmt, rb64, slug, lam=lam)
            if slug == "tr":
                resid = transpose_residual(M, w, rb64)
            else:
                resid = ridge_residual(M, w, rb64, float(lam))
            if not (resid <= RESIDUAL_RTOL):
                raise LadderHaltError(
                    f"gate (iii) FAIL at {behavior}/{slug}/L{layer:02d}: relative residual "
                    f"{resid:.3g} > {RESIDUAL_RTOL:g}"
                )
            halt_amp_tripwire(w, behavior, slug, layer)
            d = i2254.destandardized_direction(xsd, w)
            i2254._save_direction(dir_out, behavior, slug, layer, d, manifest)
            rows[f"{behavior}__{slug}"] = {
                "lambda": lam,
                "residual": resid,
                "w_norm_prenorm": float(np.linalg.norm(w)),
                "alignment_concentration": conc,
            }
            dirs[f"{behavior}__{slug}"] = d
    return {
        "layer": layer,
        "h": int(xsd.shape[0]),
        "kstar": kstar,
        "lambdas": lambdas,
        "rows": rows,
        "dirs": dirs,
        "manifest": manifest,
    }


def _tiny_bank_load(bank_root: Path, behavior: str, slug: str, layer: int) -> np.ndarray:
    """Fixture-dim mirror of ``_ensure_direction_vec`` (LOCAL ONLY, no HF
    fallback) for hidden dims != 3584. The production branch — H ==
    ``i2254.HIDDEN_DIM`` — always goes through the REAL loader; this mirror
    exists solely so tiny-H fixtures can exercise the round-trip gate
    (production-loader body coverage lives at real H in the CPU test)."""
    import torch

    path = Path(bank_root) / "directions" / f"{behavior}_{slug}_L{layer}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    vec = payload["direction"].float()
    return (vec / vec.norm()).numpy()


def phase_directions(args) -> None:
    """Phase D (plan §4.1): stage the 28 stored maps, run the HALT gates,
    build 2 × 4 × 28 = 224 directions on a ProcessPool, round-trip each saved
    file through the production loader, persist ladder_report.json, copy the
    bank files where ``_steer_hook_factory`` reads them, and upload."""
    from concurrent.futures import ProcessPoolExecutor, as_completed

    out_root = i2254._out_root(args)
    rroot = round_root(out_root)
    fk._wipe_stale_sentinels([SENTINEL_DIRECTIONS])
    i2254._assert_phase_headroom(out_root, 1.0, SENTINEL_DIRECTIONS)
    layers = sorted(int(x) for x in args.layers)
    assert layers, "directions: empty --layers"
    behaviors = tuple(b for b in args.behaviors if b in ROUND_BEHAVIORS)
    if tuple(behaviors) != ROUND_BEHAVIORS:
        raise RuntimeError(
            f"directions runs BOTH round behaviors {ROUND_BEHAVIORS} (gate (ii) spans both); "
            f"got {args.behaviors}"
        )
    maps_dir = _maps_dir(args)
    i2254._breadcrumb(SENTINEL_DIRECTIONS, layers=len(layers), behaviors=len(behaviors))

    npz_paths = {ly: _stage_map_npz(ly, maps_dir) for ly in layers}
    rb_all = _RB_LOADER()
    for b in behaviors:
        if b not in rb_all:
            raise RuntimeError(f"r_B bank missing behavior {b}")

    # Gate (ii) BEFORE the pool: rebuild the parent's d_pre from the stored W
    # at the 4 registered (behavior, layer) cells vs the committed bank files.
    parity_layers = [ly for ly in PARITY_LAYERS if ly in layers]
    if not args.smoke and sorted(parity_layers) != sorted(PARITY_LAYERS):
        raise RuntimeError(
            f"directions: production --layers must include the parity layers {PARITY_LAYERS}"
        )
    bank_root = Path(args.out_root)
    parity: dict[str, float] = {}
    for ly in parity_layers:
        z = np.load(npz_paths[ly])
        assert_npz_keys(z, ly)
        halt_npz_selfconsistency(z, ly)
        for b in behaviors:
            d_re = rebuild_parent_preimage(z, rb_all[b][ly])
            bank_vec = np.asarray(_PARENT_VEC_LOADER(bank_root, b, "pre", ly), dtype=np.float64)
            parity[f"{b}__L{ly}"] = halt_rebuild_parity(d_re, bank_vec, b, ly)
    logger.info("[%s] gate (ii) parity PASS: %s", SENTINEL_DIRECTIONS, parity)

    dir_out = rroot / "directions_ladder"
    tasks = [
        {
            "layer": ly,
            "npz_path": str(npz_paths[ly]),
            "dir_out": str(dir_out),
            "rb_rows": {b: rb_all[b][ly] for b in behaviors},
        }
        for ly in layers
    ]
    workers = max(1, min(int(args.fit_workers), len(tasks)))
    blas = max(1, (os.cpu_count() or 8) // workers)
    records: dict[int, dict] = {}
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_worker_init, initargs=(blas,)
    ) as pool:
        futs = {pool.submit(build_layer_ladder, t): t["layer"] for t in tasks}
        done = 0
        for fut in as_completed(futs):
            rec = fut.result()  # fail-fast: worker LadderHaltError propagates
            records[rec["layer"]] = rec
            done += 1
            i2254._progress(SENTINEL_DIRECTIONS, done, len(tasks), f"L{rec['layer']}", t0)

    # Copy into the parent bank dir (where `_ensure_direction_vec` /
    # `_steer_hook_factory` read), then gate (iii) loader round-trip through
    # the PRODUCTION loader at production H (tiny-H fixtures use the mirror).
    bank_dir = bank_root / "directions"
    bank_dir.mkdir(parents=True, exist_ok=True)
    n_files = 0
    for ly in layers:
        rec = records[ly]
        prod_h = rec["h"] == i2254.HIDDEN_DIM
        for key, d in rec["dirs"].items():
            b, slug = key.split("__")
            name = f"{b}_{slug}_L{ly}.pt"
            shutil.copy2(dir_out / name, bank_dir / name)
            if prod_h:
                loaded = i2254._ensure_direction_vec(bank_root, b, slug, ly).numpy()
            else:
                loaded = _tiny_bank_load(bank_root, b, slug, ly)
            rt_cos = _cos(loaded, d)
            if not (rt_cos >= LOADER_ROUNDTRIP_MIN_COS):
                raise LadderHaltError(
                    f"gate (iii) FAIL: loader round-trip cos={rt_cos!r} < "
                    f"{LOADER_ROUNDTRIP_MIN_COS!r} for {name}"
                )
            rec["rows"][key]["loader_roundtrip_cos"] = rt_cos
            n_files += 1
    expected_files = len(behaviors) * len(LADDER_SLUGS) * len(layers)
    assert n_files == expected_files, (n_files, expected_files)

    # Diagnostics (§4.1, free): cosines vs the parent's d_pre / d_ctxext / r_B
    # and vs the transpose arm — computed BEFORE any H1 narration exists.
    for ly in layers:
        rec = records[ly]
        for b in behaviors:
            d_tr = rec["dirs"][f"{b}__tr"]
            rb_u = rb_all[b][ly] / np.linalg.norm(rb_all[b][ly])
            d_pre = np.asarray(_PARENT_VEC_LOADER(bank_root, b, "pre", ly), dtype=np.float64)
            d_cxd = np.asarray(_PARENT_VEC_LOADER(bank_root, b, "ctxext", ly), dtype=np.float64)
            for slug in LADDER_SLUGS:
                d = rec["dirs"][f"{b}__{slug}"]
                row = rec["rows"][f"{b}__{slug}"]
                row["cos_vs_parent_pre"] = _cos(d, d_pre)
                row["cos_vs_ctxext"] = _cos(d, d_cxd)
                row["cos_vs_rb"] = _cos(d, rb_u)
                if slug != "tr":
                    row["cos_vs_tr"] = _cos(d, d_tr)
            rec["rows"][f"{b}__tr"]["cos_tr_vs_parent_pre"] = _cos(d_tr, d_pre)

    report = {
        "lambda_rule": (
            "lambda_j = quantile_j({Sm_i^2}) at j in {0.05, 0.50, 0.95}, full spectrum, "
            "per layer, behavior-independent (plan §4.1; np.quantile linear)"
        ),
        "gates": {
            "npz_selfconsistency": f"PASS {len(layers)}/{len(layers)}",
            "rebuild_parity_cos": parity,
            "parity_cos_min": PARITY_COS_MIN,
            "residual_rtol": RESIDUAL_RTOL,
            "loader_roundtrip_min_cos": LOADER_ROUNDTRIP_MIN_COS,
            "amp_tripwire_ratio": AMP_TRIPWIRE_RATIO,
        },
        "n_direction_files": n_files,
        "layers": {
            str(ly): {
                "kstar": records[ly]["kstar"],
                "lambdas": records[ly]["lambdas"],
                "arms": records[ly]["rows"],
            }
            for ly in layers
        },
    }
    i2254._write_json_atomic(rroot / "ladder_report.json", _ladder_metadata(report))

    # Uploads (phase-D end, BEFORE steer — §-phase-order persistence): the 224
    # .pt APPEND to the parent bank prefix (zero loader change — §4.1), the
    # report to the round prefix. ONE bulk commit each, never per-file loops.
    _UPLOAD(dir_out, f"{i2254._hf_prefix()}/directions", ["*.pt"])
    _UPLOAD(rroot, _round_hf_prefix(), ["ladder_report.json"])
    i2254._write_sentinel(
        out_root,
        SENTINEL_DIRECTIONS,
        "done",
        {"n_direction_files": n_files, "layers": len(layers)},
    )
    i2254._breadcrumb(
        SENTINEL_DIRECTIONS, status="done", files=n_files, wall_s=round(time.time() - t0, 1)
    )


# ---------------------------------------------------------------------------
# cell enumeration (§4.2: parent operating-point union × 4 arms = 44 cells)
# ---------------------------------------------------------------------------


def registered_grid_from_ops(ops: dict) -> dict[str, tuple[tuple[str, float], ...]]:
    """Derive the per-behavior (layer_config, c) union over the parent's
    context-locus operating points across {pre, rb, ctxext}, then ASSERT
    equality with the plan-registered family (drift fails loud both ways)."""
    grid: dict[str, tuple[tuple[str, float], ...]] = {}
    for b in ROUND_BEHAVIORS:
        ops_b = ops["behaviors"][b]
        pts: set[tuple[str, float]] = set()
        for d in UNION_SOURCE_ARMS:
            for breadth in i2254.BREADTHS:
                point = ops_b.get(f"{d}__context__{breadth}")
                if point is None:
                    continue
                pts.add((str(point["layer_config"]), float(point["c"])))
        derived = tuple(sorted(pts))
        registered = tuple(sorted(PLAN_GRID[b]))
        if derived != registered:
            raise RuntimeError(
                f"registered grid drift for {b}: artifact-derived {derived} != "
                f"plan-registered {registered} (plan §4.2 cell enumeration)"
            )
        grid[b] = PLAN_GRID[b]
    return grid


def registered_cells(args) -> list[dict]:
    """The 44 registered steer cells (parent cell-dict convention; smoke = the
    single plan-§4 smoke cell evil × tr × (L14, +4) — counts only, same path)."""
    if args.smoke:
        return [
            {
                "behavior": "evil",
                "kind": "steer",
                "direction": "tr",
                "position": "context",
                "layer_config": "L14",
                "c": 4.0,
            }
        ]
    ops = i2254._load_operating_points(INPUTS_ROOT)
    grid = registered_grid_from_ops(ops)
    cells: list[dict] = []
    for b in ROUND_BEHAVIORS:
        for slug in LADDER_SLUGS:
            for lc, c in grid[b]:
                cells.append(
                    {
                        "behavior": b,
                        "kind": "steer",
                        "direction": slug,
                        "position": "context",
                        "layer_config": lc,
                        "c": float(c),
                    }
                )
    assert len(cells) == FAMILY_SIZE, len(cells)
    ids = [i2254._cell_id(c) for c in cells]
    assert len(set(ids)) == FAMILY_SIZE, "duplicate cell ids in the registered family"
    return cells


def _ladder_regime_fp(args, cell: dict, rho_pooled: dict) -> str:
    """Machine-stable steer regime fingerprint (#2222/#2225 stale-cache class):
    every output-affecting dial; rho values are FILE-READ floats (never
    recomputed — the code-style float-hash rule)."""
    layers = i2254.LAYER_CONFIGS[cell["layer_config"]]
    return i2254._sha8(
        {
            "draws": int(args.draws),
            "q_steer": int(args.q_steer),
            "seeds": list(LADDER_SEEDS),
            "gen_cap": i2254.GEN_MAX_NEW_TOKENS,
            "rb_rev": i2254.HF_REV,
            "rho": {f"L{ly}": float(rho_pooled[f"L{ly}"]) for ly in layers},
        }
    )


# ---------------------------------------------------------------------------
# phase: steer (GPU; 44 cells round-robin sharded; packed HF uploads #2286)
# ---------------------------------------------------------------------------


def _upload_ladder_pack(comp_root: Path, shard_id: int, cell_names: list[str]) -> int:
    """Pack THIS SHARD's per-cell steer records into ≤9 MB JSONL line-shards
    and upload — bounded net-new file count (#2286: the shared data repo sits
    near the Hub ~1M-file ceiling; the fk `_upload_steer_pack` recipe).
    Local per-cell JSONs stay on disk (checkpoints, never deleted)."""
    import scripts.issue2220_readwrite as rw2220

    stage = comp_root.parent / f"raw_completions_stage_shard{shard_id}"
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)
    for name in cell_names:
        shutil.copy2(comp_root / name, stage / name)
    dest = comp_root.parent / f"raw_completions_pack_shard{shard_id}"
    if dest.exists():
        shutil.rmtree(dest)  # re-pack from scratch: shard numbering must not drift
    n = rw2220._pack_tree_to_jsonl_shards(
        stage, dest, group=f"ladder_steer_shard{shard_id}", pattern="*.json"
    )
    shutil.rmtree(stage)
    _UPLOAD(
        dest,
        f"{_round_hf_prefix()}/raw_completions/steer_pack/shard{shard_id}",
        ["*.jsonl", "*.json"],
    )
    return n


def phase_steer(args) -> None:
    """The 44-cell ladder grid (plan §4.2): 20 questions × 5 draws × seeds
    {42,43} per cell at the inherited operating points; per-cell JSON
    checkpoints (regime-fingerprinted cached-skip resume unless --force),
    round-robin --shard-id/--num-shards sharding (launcher pins CVD per the
    #543 recipe), cap-hit > 2% ⇒ one regen at 2× cap, packed HF raw-completion
    uploads BEFORE the shard sentinel."""
    i2254._require_cuda("steer (transpose_ladder)")
    out_root = i2254._out_root(args)
    rroot = round_root(out_root)
    fk._wipe_stale_sentinels([SENTINEL_STEER, f"{SENTINEL_STEER}-shard{args.shard_id}"])
    i2254._assert_phase_headroom(out_root, 2.0, SENTINEL_STEER)
    i2254._stage_e1_assets()
    rho_pooled, _ = i2254._load_rho(INPUTS_ROOT)
    cells = registered_cells(args)
    assert 0 <= args.shard_id < args.num_shards, (args.shard_id, args.num_shards)
    shard = cells[args.shard_id :: args.num_shards]
    comp_root = rroot / "steer" / "raw_completions"
    comp_root.mkdir(parents=True, exist_ok=True)
    i2254._breadcrumb(SENTINEL_STEER, cells=len(cells), shard=len(shard), shard_id=args.shard_id)
    if not shard:
        logger.warning(
            "[%s] shard %d/%d is EMPTY (%d cells < num_shards) — nothing to generate",
            SENTINEL_STEER,
            args.shard_id,
            args.num_shards,
            len(cells),
        )
        i2254._write_sentinel(
            out_root,
            f"{SENTINEL_STEER}-shard{args.shard_id}",
            "done",
            {"cells": 0, "regen_cells": 0, "empty_shard": True},
        )
        i2254._breadcrumb(SENTINEL_STEER, status="done", regen_cells=0, empty_shard=1)
        return

    n_pack_files = -(-len(shard) * LADDER_BYTES_PER_CELL // 9_000_000) + 1
    fk._assert_hub_headroom_for_steer(n_pack_files, len(shard) * LADDER_BYTES_PER_CELL)

    shard_names = [f"{i2254._cell_id(c)}.json" for c in shard]

    def _flush_pack() -> None:
        have = [n for n in shard_names if (comp_root / n).exists()]
        if have:
            _upload_ladder_pack(comp_root, args.shard_id, have)

    model, tok = i2254._load_model_and_tokenizer()
    bank_root = Path(args.out_root)
    q_cache = {b: i2254._eval_questions(b)[: args.q_steer] for b in ROUND_BEHAVIORS}
    for b, qs in q_cache.items():
        assert len(qs) == args.q_steer, (b, len(qs), args.q_steer)

    t0 = time.time()
    n_regen = 0
    n_generated = 0
    for k, cell in enumerate(shard, 1):
        cid = i2254._cell_id(cell)
        path = comp_root / f"{cid}.json"
        fp = _ladder_regime_fp(args, cell, rho_pooled)
        if path.exists() and not args.force:
            cached_fp = json.loads(path.read_text()).get("regime_fp")
            if cached_fp == fp:
                i2254._progress(SENTINEL_STEER, k, len(shard), f"{cid} (cached)", t0)
                continue
            logger.info(
                "[%s] %s cached record regime_fp %s != %s — cache MISS, regenerating",
                SENTINEL_STEER,
                cid,
                cached_fp,
                fp,
            )
        qs = q_cache[cell["behavior"]]
        contexts = i2254._contexts_for_questions(qs)
        q_idx = list(range(len(qs)))
        make, alphas = i2254._steer_hook_factory(model, bank_root, cell, rho_pooled)
        rec = i2254._gen_cell_rows(
            model,
            tok,
            cell,
            contexts,
            q_idx,
            make,
            n_draws=args.draws,
            seeds=LADDER_SEEDS,
            max_new_tokens=i2254.GEN_MAX_NEW_TOKENS,
            alphas=alphas,
        )
        if rec["cap_hit_fraction"] > i2254.CAP_HIT_REGEN_FRAC:
            n_regen += 1
            logger.info(
                "[%s] %s cap-hit %.3f > %.2f — regenerating at %dx cap",
                SENTINEL_STEER,
                cid,
                rec["cap_hit_fraction"],
                i2254.CAP_HIT_REGEN_FRAC,
                i2254.CAP_HIT_REGEN_FACTOR,
            )
            initial = {
                "initial_cap_hit_fraction": rec["cap_hit_fraction"],
                "initial_max_new_tokens": i2254.GEN_MAX_NEW_TOKENS,
            }
            rec = i2254._gen_cell_rows(
                model,
                tok,
                cell,
                contexts,
                q_idx,
                make,
                n_draws=args.draws,
                seeds=LADDER_SEEDS,
                max_new_tokens=i2254.GEN_MAX_NEW_TOKENS * i2254.CAP_HIT_REGEN_FACTOR,
                alphas=alphas,
            )
            rec["regen"] = initial
        rec["regime_fp"] = fp
        i2254._write_json_atomic(path, _ladder_metadata(rec))
        n_generated += 1
        if n_generated % PACK_FLUSH_EVERY == 0:  # incremental durability flush
            _flush_pack()
        i2254._progress(SENTINEL_STEER, k, len(shard), cid, t0)

    # Final pack covers the FULL shard cell set (cached cells too) so a
    # fully-cached resume still lands a complete pack before the sentinel.
    _flush_pack()
    tag = SENTINEL_STEER if args.num_shards == 1 else f"{SENTINEL_STEER}-shard{args.shard_id}"
    i2254._write_sentinel(out_root, tag, "done", {"cells": len(shard), "regen_cells": n_regen})
    i2254._breadcrumb(SENTINEL_STEER, status="done", regen_cells=n_regen)


# ---------------------------------------------------------------------------
# phase: judge (VM off-pod; Batch API; rule-26 pilot; rule-28 sync re-issue)
# ---------------------------------------------------------------------------


def _stage_ladder_completions(args, rroot: Path, expected_fp: dict[str, str]) -> Path:
    """Local-first steer raw_completions; else stage + UNPACK the per-shard
    JSONL packs (manifest-driven — un-manifested shard sets refused; duplicate
    cell paths refused; the fk `_stage_round_completions` conventions). Every
    branch ends with a regime-fp cross-check on the staged records."""
    comp_root = rroot / "steer" / "raw_completions"

    def _assert_fps(src: str) -> None:
        bad: list[str] = []
        for cid, fp in sorted(expected_fp.items()):
            p = comp_root / f"{cid}.json"
            if not p.is_file():
                continue
            got = json.loads(p.read_text()).get("regime_fp")
            if got != fp:
                bad.append(f"{cid}: {got} != {fp}")
        if bad:
            raise RuntimeError(
                f"ladder staging ({src}): {len(bad)} gen record(s) fail the regime_fp "
                f"cross-check — stale/mixed vintage refused (first: {bad[:4]})"
            )

    if comp_root.exists() and any(comp_root.glob("*.json")):
        _assert_fps("local-first")
        return comp_root
    pack_prefix = f"{_round_hf_prefix()}/raw_completions/steer_pack"
    entries = fk._hub_tree(pack_prefix, recursive=True)
    manifest_paths = sorted(e.path for e in entries if Path(e.path).name == "pack_manifest.json")
    remote_jsonl = {e.path for e in entries if e.path.endswith(".jsonl")}
    if not manifest_paths:
        raise RuntimeError(
            f"ladder judge: no steer completions locally and no pack manifests at "
            f"{pack_prefix} ({len(remote_jsonl)} un-manifested shard(s) refused)"
        )
    dl_root = rroot / "steer" / "raw_completions_pack_dl"
    seen: dict[str, str] = {}
    n_cells = 0
    for mp in manifest_paths:
        mlocal = dl_root / Path(mp).relative_to(pack_prefix)
        fk._hub_stage(mp, mlocal)
        manifest = json.loads(mlocal.read_text())
        parent = str(Path(mp).parent)
        n_rows = 0
        for name in manifest["shards"]:
            pth = f"{parent}/{name}"
            if pth not in remote_jsonl:
                raise RuntimeError(
                    f"ladder judge: manifest {mp} names shard {name} absent from the remote "
                    "listing — partial/corrupt pack upload, refusing rehydration"
                )
            local = dl_root / Path(pth).relative_to(pack_prefix)
            fk._hub_stage(pth, local)
            for line in local.open(encoding="utf-8"):
                if not line.strip():
                    continue
                row = json.loads(line)
                cname = Path(row["path"]).name
                assert cname.endswith(".json"), row["path"]
                if cname in seen:
                    raise RuntimeError(
                        f"ladder judge: duplicate cell record {cname} in {pth} (already "
                        f"unpacked from {seen[cname]}) — overlapping packs refused"
                    )
                seen[cname] = pth
                i2254._write_json_atomic(comp_root / cname, row["doc"])
                n_cells += 1
                n_rows += 1
        if n_rows != int(manifest["n_files"]):
            raise RuntimeError(
                f"ladder judge: manifest {mp} declares n_files={manifest['n_files']} but its "
                f"shards unpacked {n_rows} rows — corrupt pack refused"
            )
    if not n_cells:
        raise RuntimeError(f"ladder judge: pack manifests under {pack_prefix} unpacked ZERO cells")
    _assert_fps("pack-rehydration")
    return comp_root


def _judge_ladder_cell(args, rroot: Path, gen_path: Path, rubric: str, n_draws: int) -> dict:
    """Judge one steer cell via ``fk._judge_graded_with_refusal_reissue``
    (Batch-first + rule-28 targeted SYNC re-issue at the identical instrument;
    residual api-refusal rate fail-loud). Per-cell checkpoint at
    judge/judged/<cid>.json, resume keyed on (gen-file byte sha, judge
    instrument fingerprint) — a steer regen or instrument change is a MISS."""
    from explore_persona_space.experiments.issue_1739.constants import (
        JUDGE_MODEL,
        JUDGE_TEMPERATURE,
    )
    from explore_persona_space.experiments.issue_1739.judging import (
        judge_tallies,
        rollout_item_id,
    )

    raw = gen_path.read_bytes()
    rec = json.loads(raw)
    cell = rec["cell"]
    cid = rec["cell_id"]
    gen_sha = hashlib.sha256(raw).hexdigest()[:12]
    judge_fp = fk._judge_instrument_fp(rubric, n_draws)
    out_path = rroot / "judge" / "judged" / f"{cid}.json"
    if out_path.exists() and not args.force:
        cached = json.loads(out_path.read_text())
        if cached.get("gen_sha") == gen_sha and cached.get("judge_fp") == judge_fp:
            return cached
        logger.info(
            "[ladder-judge] %s judged checkpoint stale (gen_sha/judge_fp mismatch) — re-judging",
            cid,
        )
    qs = i2254._eval_questions(cell["behavior"])
    items: list[tuple[str, str, str]] = []
    meta: dict[str, dict] = {}
    for qi, seed, ci, di, text in i2254._iter_gen_qa(rec):
        iid = rollout_item_id(i2254._judge_ctx_id(cell, seed, len(items)), di)
        items.append((iid, qs[qi], text))
        meta[iid] = {"qi": qi, "seed": seed, "ci": ci, "di": di}
    result, merged, reissue = fk._judge_graded_with_refusal_reissue(
        items,
        rubric,
        cache_dir=rroot / "judge" / "cache" / cid,
        save_raw=rroot / "judge" / "raw" / cid,
        n_draws=n_draws,
    )
    per_q: dict[int, list[float]] = {}
    for iid, scores in merged.items():
        if scores:
            per_q.setdefault(meta[iid]["qi"], []).append(float(np.mean(scores)))
    n_q = (max(m["qi"] for m in meta.values()) + 1) if meta else 0
    per_q_mean = [float(np.mean(per_q[q])) if q in per_q else None for q in range(n_q)]
    per_q_rate = [
        float(np.mean([s >= i2254.SCORE_THRESHOLD for s in per_q[q]])) if q in per_q else None
        for q in range(n_q)
    ]
    valid_means = [m for m in per_q_mean if m is not None]
    valid_rates = [r for r in per_q_rate if r is not None]
    coherence_rate = i2254._coherence_rate(rec)
    fc_merged = (
        float(np.mean([min(len(sc), n_draws) / n_draws for sc in merged.values()]))
        if merged
        else None
    )
    out = {
        "cell_id": cid,
        "cell": cell,
        "phase": "steer",
        "gen_sha": gen_sha,
        "judge_fp": judge_fp,
        "n_questions": n_q,
        "judge": {
            "model": JUDGE_MODEL,
            "n_draws": n_draws,
            "max_tokens": i2254.JUDGE_MAX_TOKENS_2254,
            "temperature": JUDGE_TEMPERATURE,
            "transport": "batch (threshold_base=0 pin) + rule-28 sync re-issue",
        },
        "items": meta,
        "accounting": {
            **judge_tallies(result),
            "n_refusal_draws": result.n_refusal_draws,
            "n_api_refusal_draws": result.n_api_refusal_draws,
            "per_item_api_refusals": result.per_item_api_refusals,
            "frac_items_complete_batch": (result.frac_items_complete if result.scores else None),
            "frac_items_complete": fc_merged,
            "sync_reissue": reissue,
            "n_items": len(items),
            "n_items_zero_valid": sum(1 for sc in merged.values() if not sc),
        },
        "per_item_scores_merged": merged,
        "per_question_mean_score": per_q_mean,
        "per_question_rate": per_q_rate,
        "per_question_n": [len(per_q.get(q, [])) for q in range(n_q)],
        "mean_score": float(np.mean(valid_means)) if valid_means else None,
        "rate": float(np.mean(valid_rates)) if valid_rates else None,
        "coherence_rate": coherence_rate,
        "coherence_pass": bool(coherence_rate >= i2254.COHERENCE_CELL_GATE),
        "cap_hit_fraction": rec.get("cap_hit_fraction"),
        "max_new_tokens": rec.get("max_new_tokens"),
        "regen": rec.get("regen"),
        "alphas": rec.get("alphas"),
    }
    i2254._write_json_atomic(out_path, _ladder_metadata(out))
    return out


def phase_judge(args) -> None:
    """Off-pod judge wave (plan §4.3): stage/verify the 44 gen records, run the
    rule-26 pilot gate per behavior (parent instrument, truncation unwaivable),
    then the per-cell Batch wave with the rule-28 sync re-issue; rule-29
    completeness block + the git wave sentinel."""
    from explore_persona_space.experiments.issue_1739.judging import load_trait_rubric

    out_root = i2254._out_root(args)
    rroot = round_root(out_root)
    i2254._stage_e1_assets()
    rho_pooled, _ = i2254._load_rho(INPUTS_ROOT)
    cells = registered_cells(args)
    behaviors = sorted({c["behavior"] for c in cells})
    expected_fp = {i2254._cell_id(c): _ladder_regime_fp(args, c, rho_pooled) for c in cells}
    comp_root = _stage_ladder_completions(args, rroot, expected_fp)
    staged = {f.stem for f in comp_root.glob("*.json")}
    missing = sorted(set(expected_fp) - staged)
    if missing:
        raise RuntimeError(
            f"ladder judge: staged gen grid INCOMPLETE — {len(missing)} of "
            f"{len(expected_fp)} cells missing (e.g. {missing[:8]}); refusing to judge a "
            "partial family"
        )
    n_draws = i2254._judge_draws(args, "decisive")
    rubrics = {b: load_trait_rubric(b) for b in behaviors}
    for b in behaviors:
        i2254._run_judge_pilot(args, rroot, "steer", b, rubrics[b], n_draws)
    if args.pilot:
        logger.info("[ladder-judge] --pilot: rule-26 gate PASSed; stopping before the wave")
        return
    t0 = time.time()
    files = sorted(comp_root.glob("*.json"))
    for k, gen_path in enumerate(files, 1):
        j = _judge_ladder_cell(
            args,
            rroot,
            gen_path,
            rubrics[json.loads(gen_path.read_text())["cell"]["behavior"]],
            n_draws,
        )
        i2254._progress("ladder-judge", k, len(files), j["cell_id"], t0)
    judged_files = sorted((rroot / "judge" / "judged").glob("*.json"))
    i2254._write_json_atomic(
        rroot / "judge" / "completeness.json",
        _ladder_metadata(i2254._completeness_block(judged_files)),
    )
    i2254._write_json_atomic(
        rroot / "judge" / "wave_done.json",
        _ladder_metadata({"n_cells": len(judged_files), "n_draws": n_draws}),
    )
    i2254._breadcrumb("ladder-judge", status="done", cells=len(judged_files))


# ---------------------------------------------------------------------------
# phase: reduce (VM CPU; §3 verdict lattice; §6 conventions)
# ---------------------------------------------------------------------------


def _ensure_reduce_git_inputs() -> None:
    """Materialize the committed parent inputs (partial-clone pods/worktrees
    exclude other cones — #2211); fail-loud when absent."""
    for rel, cone in GIT_INPUTS:
        i2254._ensure_git_input(rel, cone)


def load_parent_floor(behavior: str) -> tuple[np.ndarray, float, float]:
    """(floor per-question means (20,), floor mean, ceiling_delta) from the
    committed baseline_ceiling artifact (plan §4.3 reduce floor; key verified
    at plan time)."""
    path = INPUTS_ROOT / "baseline_ceiling" / "judged_percell.json"
    blk = json.loads(path.read_text())["behaviors"][behavior]
    vals = blk["alpha0"]["per_question_mean_score"]
    if len(vals) != i2254.N_EVAL_QUESTIONS or any(v is None for v in vals):
        raise RuntimeError(
            f"parent floor for {behavior}: expected {i2254.N_EVAL_QUESTIONS} non-null "
            f"per-question means, got {vals!r}"
        )
    return (
        np.asarray(vals, dtype=np.float64),
        float(blk["alpha0"]["mean_score"]),
        float(blk["ceiling_delta"]),
    )


def load_parent_band(behavior: str) -> float:
    """Parent context-locus null band p97.5 — READ from the committed
    decisive/verdicts.json (never hardcoded; plan §12.9)."""
    path = INPUTS_ROOT / "decisive" / "verdicts.json"
    band = json.loads(path.read_text())["behaviors"][behavior]["null_band_context"]
    return float(band["p975"])


def assert_parent_reference_margin() -> dict:
    """§12.19 fixture (runs BEFORE any new cell enters the reduce): reproduce
    the parent's published evil measured-context-direction margin
    (evil cxd L14 c4 → 2.4586) from parent artifacts through THIS round's
    margin convention (margin = Δ − band, band read from the artifact)."""
    percell = json.loads((INPUTS_ROOT / "decisive" / "delta_score_percell.json").read_text())[
        "behaviors"
    ]["evil"]
    delta = float(percell[PARENT_REFERENCE_CELL]["delta_score"])
    band = load_parent_band("evil")
    margin = delta - band
    published = json.loads((INPUTS_ROOT / "decisive" / "verdicts.json").read_text())["behaviors"][
        "evil"
    ]["margins"]["E_ctxdir"]
    if abs(margin - PARENT_REFERENCE_MARGIN) > 1e-9:
        raise RuntimeError(
            f"§12.19 parent-reference fixture FAIL: recomputed margin {margin!r} != "
            f"published {PARENT_REFERENCE_MARGIN!r} for {PARENT_REFERENCE_CELL}"
        )
    if (
        published["cell_id"] != PARENT_REFERENCE_CELL
        or abs(float(published["value"]) - margin) > 1e-9
    ):
        raise RuntimeError(
            f"§12.19 parent-reference fixture FAIL: verdicts.json E_ctxdir "
            f"{published!r} does not match the recomputed margin {margin!r}"
        )
    return {"cell_id": PARENT_REFERENCE_CELL, "margin": margin, "verdict": "PASS"}


def _boot_diffs(cell_q: np.ndarray, ref_q: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Per-draw paired delta vector under one shared index matrix (the
    ``_boot_diff_ci`` formula, kept for tag quantiles + selection curves; a
    consistency assert against ``_boot_diff_ci`` guards drift)."""
    return np.nanmean(cell_q[idx], axis=1) - np.nanmean(ref_q[idx], axis=1)


def _intrusion_flags(gen_rec: dict, rx, tok) -> dict[tuple[int, int], bool]:
    """Per-(ci, di) CJK flag on the common 2048-token horizon (the r5
    ``issue2254_firstk_ctxext_sensitivity`` convention; parent committed
    regex). Completion text never logged."""
    flags: dict[tuple[int, int], bool] = {}
    for _qi, _seed, ci, di, text in i2254._iter_gen_qa(gen_rec):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        t_common = (
            text if len(ids) <= COMMON_HORIZON_TOKENS else tok.decode(ids[:COMMON_HORIZON_TOKENS])
        )
        flags[(ci, di)] = bool(rx.search(t_common))
    return flags


def _intrusion_sensitivity(judged: dict, gen_rec: dict, rx, tok, floor_mean: float) -> dict:
    """Zeroed/excluded intrusion sensitivity reads for one cell (SENSITIVITY
    only — the BINDING headline scores intruded rows AS JUDGED, §6). Cell
    aggregate = unweighted mean of per-question means over questions with ≥1
    valid row (the stored ``mean_score`` convention, replay-asserted)."""
    flags = _intrusion_flags(gen_rec, rx, tok)
    n_total = len(flags)
    cjk_frac = (sum(flags.values()) / n_total) if n_total else 0.0
    rows: list[tuple[int, float, bool]] = []
    for rid, m in judged["items"].items():
        draws = judged["per_item_scores_merged"].get(rid) or []
        vals = [float(v) for v in draws if isinstance(v, int | float)]
        if not vals:
            continue  # judge-dropped row — outside the mean by design
        rows.append((m["qi"], float(np.mean(vals)), flags[(m["ci"], m["di"])]))
    if not rows:
        return {"cjk_common": cjk_frac, "n_valid_judge_rows": 0, "undefined": True}
    qis = np.array([q for q, _, _ in rows])
    means = np.array([v for _, v, _ in rows])
    intr = np.array([f for _, _, f in rows])

    def _cell_mean(row_means: np.ndarray, row_qis: np.ndarray) -> float | None:
        qmeans = [float(row_means[row_qis == q].mean()) for q in np.unique(row_qis)]
        return float(np.mean(qmeans)) if qmeans else None

    mean_asis = _cell_mean(means, qis)
    stored = judged["mean_score"]
    if not (mean_asis is not None and abs(mean_asis - float(stored)) < 1e-9):
        raise RuntimeError(
            f"intrusion sensitivity replay FAIL for {judged['cell_id']}: as-is mean "
            f"{mean_asis!r} != stored mean_score {stored!r}"
        )
    mean_zeroed = _cell_mean(np.where(intr, 0.0, means), qis)
    mean_excl = _cell_mean(means[~intr], qis[~intr]) if (~intr).any() else None
    return {
        "cjk_common": cjk_frac,
        "n_valid_judge_rows": int(len(rows)),
        "n_intruded_valid": int(intr.sum()),
        "mean_asis": mean_asis,
        "mean_zeroed_intrusion": mean_zeroed,
        "mean_excluded_intrusion": mean_excl,
        "delta_zeroed_intrusion": mean_zeroed - floor_mean,
        "delta_excluded_intrusion": None if mean_excl is None else mean_excl - floor_mean,
        "convention": "binding read scores intruded rows AS JUDGED; these are sensitivity reads",
    }


def phase_reduce(args) -> None:
    """§3/§6 reduce (VM CPU, git-committed outputs): per-cell paired Δ vs the
    parent floor (1,000-draw frozen CIs), 2,000-draw verdict margins vs the
    parent band, the Undefined-cell rule, two-grain Bonferroni tags (/44
    family, /11 within-arm), per-arm + all-44 selection-aware companions,
    intrusion/cap-hit sensitivity reads, and the §3 H1/H2 lattice with
    ``fresh_nulls: false``. The §12.19 parent-reference fixture runs FIRST."""
    import re

    out_root = i2254._out_root(args)
    rroot = round_root(out_root)
    _ensure_reduce_git_inputs()
    fixture = assert_parent_reference_margin()
    logger.info("[ladder-reduce] §12.19 parent-reference fixture: %s", fixture)

    cells = registered_cells(args)
    jd = rroot / "judge" / "judged"
    comp_root = rroot / "steer" / "raw_completions"
    rx = re.compile(json.loads((INPUTS_ROOT / "decisive" / "cjk_audit.json").read_text())["regex"])
    tok = _TOKENIZER_LOADER()

    floors = {b: load_parent_floor(b) for b in ROUND_BEHAVIORS}
    bands = {b: load_parent_band(b) for b in ROUND_BEHAVIORS}

    percell: dict = {"behaviors": {b: {} for b in ROUND_BEHAVIORS}}
    defined: list[tuple[str, str, np.ndarray]] = []  # (cid, behavior, cell_q)
    undefined_cells: list[str] = []
    cell_rows: dict[str, dict] = {}
    for cell in cells:
        cid = i2254._cell_id(cell)
        b = cell["behavior"]
        jpath = jd / f"{cid}.json"
        if not jpath.is_file():
            raise FileNotFoundError(
                f"reduce: judged cell {cid} missing under {jd} — incomplete judge wave "
                "(run --phases judge first; never reduce a partial family)"
            )
        judged = json.loads(jpath.read_text())
        gen_rec = json.loads((comp_root / f"{cid}.json").read_text())
        floor_q, floor_mean, _ceiling = floors[b]
        band = bands[b]
        cell_q = i2254._q_arr(judged)
        if len(cell_q) != len(floor_q):
            if not args.smoke:
                raise RuntimeError(
                    f"reduce: {cid} has {len(cell_q)} questions vs floor "
                    f"{len(floor_q)} — production refuses a truncated grain"
                )
            floor_q = floor_q[: len(cell_q)]  # smoke-only truncation (counts, not code path)
        sens = _intrusion_sensitivity(judged, gen_rec, rx, tok, floor_mean)
        row: dict = {
            "cell": cell,
            "coherence_pass": judged["coherence_pass"],
            "coherence_rate": judged["coherence_rate"],
            "frac_items_complete": judged["accounting"]["frac_items_complete"],
            "cap_hit_fraction": judged.get("cap_hit_fraction"),
            "band_p975": band,
            "sensitivity": sens,
        }
        if np.all(np.isnan(cell_q)):
            # Undefined-cell rule (§3, registered): zero scored rows — no
            # margin_lo; OUTSIDE both H1 support and bounded-non-clear
            # narration; never evidence of misalignment.
            row.update(
                {
                    "delta_score": None,
                    "margin": None,
                    "margin_lo": None,
                    "label": "Undefined (no valid measurement)",
                }
            )
            undefined_cells.append(cid)
        else:
            idx_cell = i2254._boot_idx(len(floor_q), i2254.N_BOOT_CELL, cid + "__ladder_cell")
            point, lo_c, hi_c = i2254._boot_diff_ci(cell_q, floor_q, idx_cell)
            idx_v = i2254._boot_idx(len(floor_q), i2254.N_BOOT_VERDICT, cid + "__ladder_verdict")
            point_v, lo_v, hi_v = i2254._boot_diff_ci(cell_q, floor_q, idx_v)
            diffs_v = _boot_diffs(cell_q, floor_q, idx_v)
            # drift guard: the local per-draw vector must reproduce the parent
            # helper's quantiles exactly (same formula, same index matrix).
            assert abs(float(np.nanquantile(diffs_v, 0.025)) - lo_v) < 1e-12
            assert abs(float(np.nanquantile(diffs_v, 0.975)) - hi_v) < 1e-12
            tag_family_lo = float(np.nanquantile(diffs_v, ALPHA / (2.0 * FAMILY_SIZE)))
            tag_arm_lo = float(np.nanquantile(diffs_v, ALPHA / (2.0 * ARM_SIZE)))
            row.update(
                {
                    "delta_score": point,
                    "ci_frozen": [lo_c, hi_c],
                    "ci_label": f"frozen (registered family, n_q={len(floor_q)})",
                    "margin": point_v - band,
                    "margin_ci_verdict": [lo_v - band, hi_v - band],
                    "margin_lo": lo_v - band,
                    "margin_hi": hi_v - band,
                    "clears_nominal": bool(lo_v - band > 0.0),
                    "tags": {
                        "family_bonferroni_alpha": ALPHA / FAMILY_SIZE,
                        "multiplicity_robust_family": bool(tag_family_lo - band > 0.0),
                        "within_arm_alpha": ALPHA / ARM_SIZE,
                        "multiplicity_robust_within_arm": bool(tag_arm_lo - band > 0.0),
                        "granularity_note": (
                            f"{i2254.N_BOOT_VERDICT} bootstrap draws resolve p at "
                            "~0.0005 granularity near the /44 threshold (§6)"
                        ),
                    },
                }
            )
            defined.append((cid, b, cell_q))
        percell["behaviors"][b][cid] = row
        cell_rows[cid] = row

    # Selection-aware companions (§6): re-argmax per bootstrap draw over an
    # arm's tested cells (11), per (behavior, arm) for the hero whiskers, and
    # over ALL 44 registered cells for the overall-winner companion. Margins
    # (band-subtracted) so cells are comparable across behaviors.
    def _sel_aware(entries: list[tuple[str, str, np.ndarray]], seed_key: str) -> dict | None:
        if not entries:
            return None
        nq = len(entries[0][2])
        idx = i2254._boot_idx(nq, i2254.N_BOOT_VERDICT, seed_key)
        per_draw = np.stack(
            [_boot_diffs(cq, floors[b][0][:nq], idx) - bands[b] for (_cid, b, cq) in entries],
            axis=1,
        )
        maxes = np.nanmax(per_draw, axis=1)
        pts = [cell_rows[cid]["margin"] for (cid, _b, _cq) in entries]
        best = int(np.nanargmax(pts))
        return {
            "argmax_cell": entries[best][0],
            "point_margin": float(pts[best]),
            "ci": [float(np.nanquantile(maxes, 0.025)), float(np.nanquantile(maxes, 0.975))],
            "n_cells": len(entries),
            "n_draws": i2254.N_BOOT_VERDICT,
            "convention": "re-argmax per bootstrap draw (parent selection_inherited analogue)",
        }

    selection_aware: dict = {"arm": {}, "behavior_arm": {}}
    for slug in LADDER_SLUGS:
        ent = [
            (cid, b, cq) for (cid, b, cq) in defined if cell_rows[cid]["cell"]["direction"] == slug
        ]
        selection_aware["arm"][slug] = _sel_aware(ent, f"ladder__{slug}__selaware")
        for b in ROUND_BEHAVIORS:
            ent_b = [(cid, bb, cq) for (cid, bb, cq) in ent if bb == b]
            selection_aware["behavior_arm"][f"{b}__{slug}"] = _sel_aware(
                ent_b, f"ladder__{b}__{slug}__selaware"
            )
    all44 = _sel_aware(defined, "ladder__all44__selaware")

    # §3 lattice: H1 ⇔ margin_lo > 0 in ≥1 registered cell (coherence-gated);
    # H2 = the literal complement over DEFINED cells; all-Undefined = a
    # measurement failure, not H2.
    clearing = [
        cid
        for (cid, _b, _cq) in defined
        if cell_rows[cid]["clears_nominal"] and cell_rows[cid]["coherence_pass"]
    ]
    bounded_nonclear = [cid for (cid, _b, _cq) in defined if cell_rows[cid]["margin_hi"] <= 0.0]
    straddling = [
        cid
        for (cid, _b, _cq) in defined
        if cell_rows[cid]["margin_lo"] <= 0.0 < cell_rows[cid]["margin_hi"]
    ]
    if not defined:
        label = "Undefined (measurement failure — all registered cells undefined)"
    elif clearing:
        label = "H1"
    else:
        label = "H2"
    verdicts = {
        "label": label,
        "fresh_nulls": False,
        "inference_scope_note": (
            "band/floor/ceiling are REUSED parent artifacts measured for OTHER directions "
            "at matched injected norm; no fresh nulls were run — clears are read against a "
            "reused scalar reference band (plan §5 scope-mandated caveat)"
        ),
        "bands": bands,
        "band_source": "eval_results/issue_2254/decisive/verdicts.json null_band_context.p975",
        "floor_source": (
            "eval_results/issue_2254/baseline_ceiling/judged_percell.json "
            "behaviors.<b>.alpha0.per_question_mean_score"
        ),
        "registered_family": {
            "n_cells": FAMILY_SIZE,
            "n_per_arm": ARM_SIZE,
            "grid": {b: [list(p) for p in PLAN_GRID[b]] for b in ROUND_BEHAVIORS},
        },
        "parent_reference_margin_check": fixture,
        "h1_clearing_cells": clearing,
        "n_clearing": len(clearing),
        "narration": {
            "bounded_nonclear_cells": bounded_nonclear,
            "straddling_cells": straddling,
            "undefined_cells": undefined_cells,
            "rule": (
                "bounded non-clears (CI_hi <= band) are evidence against clearing at that "
                "cell; straddles are noise-limited (no verdict); an all-straddle world is "
                "'indistinguishable from the band given the variance' (§6 narration rules)"
            ),
        },
        "selection_aware": selection_aware,
        "all44_companion": all44,
        "bootstrap": {
            "n_cell": i2254.N_BOOT_CELL,
            "n_verdict": i2254.N_BOOT_VERDICT,
            "seed": i2254.BOOTSTRAP_SEED,
            "clustering": "question-level paired cluster bootstrap (parent convention)",
        },
        "cells": {
            cid: {
                k: row[k]
                for k in (
                    "margin",
                    "margin_lo",
                    "margin_hi",
                    "clears_nominal",
                    "coherence_pass",
                    "label",
                    "tags",
                )
                if k in row
            }
            for cid, row in cell_rows.items()
        },
    }
    i2254._write_json_atomic(
        rroot / "reduce" / "delta_score_percell.json", _ladder_metadata(percell)
    )
    i2254._write_json_atomic(rroot / "reduce" / "verdicts.json", _ladder_metadata(verdicts))
    i2254._breadcrumb(
        "ladder-reduce",
        status="done",
        label=label,
        clearing=len(clearing),
        undefined=len(undefined_cells),
    )


# ---------------------------------------------------------------------------
# phase: figures (VM CPU; scripts/issue2254_ladder_figures.py)
# ---------------------------------------------------------------------------


def phase_figures(args) -> None:
    """Plan §6 figures: the pinv→transpose hero ladder + the exploratory dump,
    rendered from the reduce's committed JSONs via
    ``scripts.issue2254_ladder_figures`` (PNG + .meta.json sidecars carrying
    the ``fresh_nulls: false`` scope note; figures stay caption-free)."""
    import scripts.issue2254_ladder_figures as ladder_figs

    out_root = Path(args.out_root)
    rroot = round_root(out_root)
    fk._wipe_stale_sentinels([SENTINEL_FIGURES])
    fig_dir = (
        Path(args.fig_dir)
        if args.fig_dir
        else _REPO_ROOT / "figures" / "issue_2254" / FOLLOWUP_LABEL
    )
    require = () if args.smoke else ladder_figs.REQUIRED_FIGURES
    res = ladder_figs.render_all(rroot, fig_dir, require=require)
    logger.info("[%s] rendered=%s skipped=%s", SENTINEL_FIGURES, res["rendered"], res["skipped"])
    i2254._write_json_atomic(
        fig_dir / "figures_manifest.json",
        _ladder_metadata({"followup_label": FOLLOWUP_LABEL, **res}),
    )
    i2254._write_sentinel(out_root, SENTINEL_FIGURES, "done", {"rendered": len(res["rendered"])})
    i2254._breadcrumb(SENTINEL_FIGURES, status="done", rendered=len(res["rendered"]))


# ---------------------------------------------------------------------------
# --rig-health (plan §4 smoke item (d)): ADVISORY probe, never a verdict input
# ---------------------------------------------------------------------------


def run_rig_health(args) -> None:
    """Advisory rig-health probe (plan §4 smoke (d)): sycophancy × parent
    ctxext × (mid, +4) × 5 questions × 2 draws through the production
    generate + judge path, expecting visibly elevated scores vs the parent
    floor. Deliberately unpowered; logs + persists ONLY (a diagnostic stop
    trigger per §7(d), never a reported result)."""
    from explore_persona_space.experiments.issue_1739.judging import load_trait_rubric

    i2254._require_cuda("rig-health probe")
    out_root = i2254._out_root(args)
    rroot = round_root(out_root)
    i2254._stage_e1_assets()
    rho_pooled, _ = i2254._load_rho(INPUTS_ROOT)
    cell = {
        "behavior": "sycophancy",
        "kind": "steer",
        "direction": "ctxext",
        "position": "context",
        "layer_config": "mid",
        "c": 4.0,
    }
    model, tok = i2254._load_model_and_tokenizer()
    qs = i2254._eval_questions("sycophancy")[:5]
    contexts = i2254._contexts_for_questions(qs)
    make, alphas = i2254._steer_hook_factory(model, Path(args.out_root), cell, rho_pooled)
    rec = i2254._gen_cell_rows(
        model,
        tok,
        cell,
        contexts,
        list(range(len(qs))),
        make,
        n_draws=2,
        seeds=(LADDER_SEEDS[0],),
        max_new_tokens=i2254.GEN_MAX_NEW_TOKENS,
        alphas=alphas,
    )
    items = []
    for qi, seed, _ci, di, text in i2254._iter_gen_qa(rec):
        items.append((f"righealth-s{seed}-q{qi:02d}-d{di}", qs[qi], text))
    result, merged, _reissue = fk._judge_graded_with_refusal_reissue(
        items,
        load_trait_rubric("sycophancy"),
        cache_dir=rroot / "smoke" / "rig_health_cache",
        save_raw=rroot / "smoke" / "rig_health_raw",
        n_draws=2,
    )
    row_means = [float(np.mean(sc)) for sc in merged.values() if sc]
    if not row_means:
        raise RuntimeError("rig-health probe: zero scored rows — rig broken, diagnose (§7(d))")
    mean = float(np.mean(row_means))
    _floor_q, floor_mean, _ceiling = load_parent_floor("sycophancy")
    out = {
        "cell": cell,
        "n_scored_rows": len(row_means),
        "mean_score": mean,
        "floor_mean": floor_mean,
        "delta_vs_floor": mean - floor_mean,
        "advisory": (
            "ADVISORY ONLY (plan §7(d)): an on-floor read here flags a broken rig before "
            "the grid — a diagnostic stop trigger, never a verdict input; deliberately "
            "unpowered and never reported as a result"
        ),
        "n_total_draws": result.n_total_draws,
    }
    i2254._write_json_atomic(rroot / "smoke" / "rig_health.json", _ladder_metadata(out))
    i2254._breadcrumb(
        "ladder-rig-health",
        status="done",
        mean=round(mean, 2),
        floor=round(floor_mean, 2),
        delta=round(mean - floor_mean, 2),
    )


# ---------------------------------------------------------------------------
# CPU-smoke fixtures (shared with tests/test_issue2254_transpose_ladder.py)
# ---------------------------------------------------------------------------

CPU_SMOKE_SCRATCH = Path("/tmp/issue-2254-ladder-cpusmoke")
FIXTURE_H = 16
FIXTURE_LAYERS = (14, 17)


def make_fixture_maps(maps_dir: Path, layers=FIXTURE_LAYERS, h: int = FIXTURE_H, seed: int = 0):
    """Synthetic per-layer map npzs with the EXACT production key set, fitted
    through the parent's own ``ridge_fit_matrix`` (so gate (i) holds by
    construction), plus a synthetic r_B bank {behavior: (28, h)}."""
    rng = np.random.default_rng(seed)
    maps_dir.mkdir(parents=True, exist_ok=True)
    rb = {b: rng.normal(size=(i2254.N_LAYERS, h)) for b in ROUND_BEHAVIORS}
    for ly in layers:
        x = rng.normal(size=(64, h))
        y = x @ rng.normal(size=(h, h)) * 0.5 + rng.normal(size=(64, h)) * 0.1
        fit = i2254.ridge_fit_matrix(x, y)
        np.savez(
            maps_dir / f"L{ly:02d}.npz",
            W=fit["W"].astype(np.float32),
            xmu=fit["xmu"],
            xsd=fit["xsd"],
            ymu=fit["ymu"],
            s=fit["s"],
            lam=np.float64(fit["lam"]),
            kstar=np.int64(i2254.kstar_from_fit(fit["s"], fit["lam"])),
            n_rows=np.int64(64),
            pass_b_revision=np.bytes_(i2254.HF_REV.encode()),
        )
    return rb


def make_fixture_parent_bank(bank_root: Path, maps_dir: Path, rb: dict, layers=FIXTURE_LAYERS):
    """Committed-bank stand-ins: d_pre + a ctxext stand-in per (behavior,
    layer), built through the parent rebuild path so gate (ii) parity holds."""
    bank_dir = Path(bank_root) / "directions"
    bank_dir.mkdir(parents=True, exist_ok=True)
    manifest: list = []
    for ly in layers:
        z = np.load(maps_dir / f"L{ly:02d}.npz")
        for b in ROUND_BEHAVIORS:
            d_pre = rebuild_parent_preimage(z, rb[b][ly])
            i2254._save_direction(bank_dir, b, "pre", ly, d_pre, manifest)
            rng = np.random.default_rng(1000 + ly)
            cx = rng.normal(size=d_pre.shape[0])
            i2254._save_direction(bank_dir, b, "ctxext", ly, cx / np.linalg.norm(cx), manifest)


def _fixture_parent_vec_loader(bank_root: Path, behavior: str, slug: str, layer: int):
    """Seam stand-in for ``_parent_vec_canonical``: local fixture bank only
    (no HF), tiny-H tolerant; same load semantics (unit-norm fp32)."""
    return _tiny_bank_load(Path(bank_root), behavior, slug, layer)


def _fixture_upload(local_dir: Path, path_in_repo: str, allow=None) -> None:
    """Seam stand-in for ``_upload_folder_to_hf``: records the planned upload
    (no network) — LOUD, never silent."""
    logger.info("[cpu-smoke] upload SKIPPED (fixture seam): %s -> %s", local_dir, path_in_repo)


class _FixtureTokenizer:
    """Whitespace tokenizer stand-in mirroring the two call shapes the
    intrusion recount uses (``tok(text, add_special_tokens=False)`` +
    ``tok.decode``)."""

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        toks = text.split(" ")
        return {"input_ids": list(range(len(toks)))}

    def decode(self, ids) -> str:
        return " ".join("w" for _ in ids)


def make_fixture_round(rroot: Path, args, deltas: dict[str, float] | None = None) -> list[dict]:
    """Synthetic judged + gen records for every registered cell (constant
    per-question deltas vs the COMMITTED parent floor ⇒ point CIs, exact
    arithmetic). One evil cell is all-dropped (the Undefined-cell rule leg);
    one sycophancy cell carries a CJK-intruded completion (intrusion leg)."""
    cells = registered_cells(args)
    jd = rroot / "judge" / "judged"
    comp = rroot / "steer" / "raw_completions"
    jd.mkdir(parents=True, exist_ok=True)
    comp.mkdir(parents=True, exist_ok=True)
    undefined_cid = "evil__tr__ctx__all__c0p5"
    intruded_cid = "sycophancy__tr__ctx__mid__c4"
    clear_cid = "sycophancy__tr__ctx__L17__c4"
    for cell in cells:
        cid = i2254._cell_id(cell)
        b = cell["behavior"]
        floor_q, _fm, _cd = load_parent_floor(b)
        n_q = len(floor_q)
        if deltas and cid in deltas:
            delta = deltas[cid]
        elif cid == clear_cid:
            delta = 30.0  # clears the sycophancy band (10.89) decisively
        elif b == "evil":
            delta = 0.0  # evil band is exactly 0: margin_lo == 0 must NOT clear (strict >)
        else:
            delta = 1.0  # bounded non-clear vs the sycophancy band (10.89)
        undefined = cid == undefined_cid
        texts = ["a plain fixture answer" for _ in range(n_q)]
        if cid == intruded_cid:
            texts[0] = "a plain fixture answer 好"
        gen = {
            "cell_id": cid,
            "cell": cell,
            "alphas": {"L14": 0.1},
            "q_of_context": list(range(n_q)),
            "seeds": {
                "42": {
                    "completions": [[t] for t in texts],
                    "coherent_flags": [[True] for _ in texts],
                    "condition_passes": [True for _ in texts],
                }
            },
            "max_new_tokens": i2254.GEN_MAX_NEW_TOKENS,
            "cap_hit_fraction": 0.0,
        }
        i2254._write_json_atomic(comp / f"{cid}.json", gen)
        items = {}
        merged = {}
        pq_mean: list[float | None] = []
        for qi in range(n_q):
            iid = f"{cid}-q{qi:02d}-r0"
            items[iid] = {"qi": qi, "seed": 42, "ci": qi, "di": 0}
            if undefined:
                merged[iid] = []
                continue
            merged[iid] = [float(floor_q[qi] + delta)]
        pq_mean = [None if undefined else float(floor_q[qi] + delta) for qi in range(n_q)]
        valid = [m for m in pq_mean if m is not None]
        judged = {
            "cell_id": cid,
            "cell": cell,
            "phase": "steer",
            "n_questions": n_q,
            "judge": {"model": "fixture", "n_draws": 1, "max_tokens": 2048, "temperature": 1.0},
            "items": items,
            "accounting": {"frac_items_complete": 0.0 if undefined else 1.0},
            "per_item_scores_merged": merged,
            "per_question_mean_score": pq_mean,
            "per_question_rate": [
                None if m is None else float(m >= i2254.SCORE_THRESHOLD) for m in pq_mean
            ],
            "per_question_n": [0 if undefined else 1 for _ in range(n_q)],
            "mean_score": float(np.mean(valid)) if valid else None,
            "rate": None,
            "coherence_rate": 1.0,
            "coherence_pass": True,
            "cap_hit_fraction": 0.0,
        }
        i2254._write_json_atomic(jd / f"{cid}.json", judged)
    return cells


def run_cpu_smoke(args) -> None:
    """VM CPU smoke (no GPU / no API / no HF writes): (a) the REAL
    ``phase_directions`` on tiny-H fixtures (all three HALT gates on the
    positive path + one negative probe each for gates (i)/(ii)); (b) the REAL
    ``phase_reduce`` on a synthetic full-44 fixture round against the
    COMMITTED parent floor/band artifacts (§12.19 fixture included); (c) the
    REAL ``phase_figures`` on that output. Module seams rebound within
    try/finally (disclosed in the module docstring)."""
    global _RB_LOADER, _PARENT_VEC_LOADER, _UPLOAD, _TOKENIZER_LOADER
    t0 = time.time()
    scratch = CPU_SMOKE_SCRATCH
    if scratch.exists():
        shutil.rmtree(scratch)
    maps_dir = scratch / "maps" / "perlayer"
    rb = make_fixture_maps(maps_dir)
    make_fixture_parent_bank(scratch, maps_dir, rb)
    ns = argparse.Namespace(**vars(args))
    ns.out_root = str(scratch)
    ns.maps_dir = str(maps_dir)
    ns.layers = list(FIXTURE_LAYERS)
    ns.behaviors = list(ROUND_BEHAVIORS)
    ns.smoke = True  # fixture grain; the registered-cell family stays full below
    ns.fit_workers = 2
    keep = (_RB_LOADER, _PARENT_VEC_LOADER, _UPLOAD, _TOKENIZER_LOADER)
    evidence: dict = {}
    try:
        _RB_LOADER = lambda: rb  # noqa: E731
        _PARENT_VEC_LOADER = _fixture_parent_vec_loader
        _UPLOAD = _fixture_upload
        _TOKENIZER_LOADER = _FixtureTokenizer
        phase_directions(ns)
        rroot = round_root(Path(ns.out_root))
        report = json.loads((rroot / "ladder_report.json").read_text())
        evidence["directions"] = {
            "n_direction_files": report["n_direction_files"],
            "parity": report["gates"]["rebuild_parity_cos"],
        }
        # negative probes: gate (i) corrupted kstar; gate (ii) parity mismatch
        z = dict(np.load(maps_dir / "L14.npz"))
        z["kstar"] = np.int64(int(z["kstar"]) + 7)
        try:
            halt_npz_selfconsistency(z, 14)
            raise AssertionError("gate (i) negative probe did NOT raise")
        except LadderHaltError:
            evidence["gate_i_negative_probe"] = "raised as designed"
        d_re = rebuild_parent_preimage(np.load(maps_dir / "L14.npz"), rb["evil"][14])
        perturbed = np.roll(d_re, 3)
        try:
            halt_rebuild_parity(d_re, perturbed, "evil", 14)
            raise AssertionError("gate (ii) negative probe did NOT raise")
        except LadderHaltError:
            evidence["gate_ii_negative_probe"] = "raised as designed"
        # (b) reduce on a synthetic FULL-44 fixture round (production grain).
        ns2 = argparse.Namespace(**vars(ns))
        ns2.smoke = False  # full registered family; committed floor grain (20q)
        make_fixture_round(rroot, ns2)
        phase_reduce(ns2)
        verdicts = json.loads((rroot / "reduce" / "verdicts.json").read_text())
        evidence["reduce"] = {
            "label": verdicts["label"],
            "n_clearing": verdicts["n_clearing"],
            "fresh_nulls": verdicts["fresh_nulls"],
            "undefined_cells": verdicts["narration"]["undefined_cells"],
        }
        # (c) figures on the fixture reduce output.
        ns2.fig_dir = str(scratch / "figures")
        phase_figures(ns2)
        evidence["figures"] = json.loads((Path(ns2.fig_dir) / "figures_manifest.json").read_text())[
            "rendered"
        ]
    finally:
        _RB_LOADER, _PARENT_VEC_LOADER, _UPLOAD, _TOKENIZER_LOADER = keep
    out_dir = Path(args.cpu_smoke_out)
    i2254._write_json_atomic(out_dir / "cpu_smoke_ladder.json", _ladder_metadata(evidence))
    i2254._breadcrumb(
        "ladder-cpu-smoke",
        status="done",
        files=evidence["directions"]["n_direction_files"],
        label=evidence["reduce"]["label"],
        figures=len(evidence["figures"]),
        elapsed=f"{time.time() - t0:.0f}s",
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

PHASES = {
    "directions": phase_directions,
    "steer": phase_steer,
    "judge": phase_judge,
    "reduce": phase_reduce,
    "figures": phase_figures,
}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="issue #2254 follow-up: transpose_ladder (plan v14) — forward-weighted "
        "map pullbacks at the context vector"
    )
    ap.add_argument(
        "--phases",
        default=None,
        help="comma-separated phases in order (directions,steer,judge,reduce,figures)",
    )
    ap.add_argument("--behaviors", nargs="+", default=list(ROUND_BEHAVIORS))
    ap.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=list(i2254.ALL_LAYERS),
        help="direction-construction layers (production: all 28; smoke: 14 17)",
    )
    ap.add_argument(
        "--out-root",
        default="eval_results/issue_2254",
        help=(
            "ISSUE out-root (parent convention); round outputs land under "
            f"<out-root>/{FOLLOWUP_LABEL}/ — reused inputs resolve at canonical "
            "committed locations independent of this flag"
        ),
    )
    ap.add_argument(
        "--maps-dir",
        default=None,
        help="stored per-layer map npz dir (default <out-root>/maps/perlayer; HF-staged)",
    )
    ap.add_argument(
        "--fit-workers",
        type=int,
        default=8,
        help="ProcessPool width for the 28 per-layer SVDs (plan §9; BLAS capped per worker)",
    )
    ap.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="round-robin cell shard (launcher pins CUDA_VISIBLE_DEVICES per shard, #543)",
    )
    ap.add_argument("--num-shards", type=int, default=1, help="total steer shards (plan §9: 4)")
    ap.add_argument(
        "--q-steer", type=int, default=Q_STEER_DEFAULT, help="eval questions per cell (§4.2: 20)"
    )
    ap.add_argument(
        "--draws",
        type=int,
        default=DRAWS_DEFAULT,
        help="gen draws per question per seed (§4.2: 5; seeds fixed at {42,43})",
    )
    ap.add_argument(
        "--pilot",
        action="store_true",
        help="judge phase: run the rule-26 pilot gate and STOP before the 44k wave",
    )
    ap.add_argument(
        "--waive-judge-parse-fail-arms",
        nargs="*",
        default=[],
        help=(
            "rule 26(b) explained-content-drop escape: pilot arm names whose parse-fail "
            "check is waived (truncation FAIL stays unwaivable inside judge_pilot)"
        ),
    )
    ap.add_argument("--force", action="store_true", help="ignore per-cell checkpoint caches")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "tiny slice (plan §4 smoke (c)): evil × tr × (L14,+4), 2q × 2 draws, layers "
            "{14,17}; scratch out-root + smoke/ HF sub-prefix (inputs stay canonical)"
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="enumerate the phase grid + resolve deferred imports, no GPU/HF/model",
    )
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="AST arg-attribute completeness + helper-call bind check, then exit 0",
    )
    ap.add_argument(
        "--fig-dir",
        default=None,
        help=f"figures dir (default figures/issue_2254/{FOLLOWUP_LABEL}/; smoke rebinds)",
    )
    ap.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="VM smoke (no GPU/API/HF writes): fixture directions + reduce + figures "
        "through the real phase entrypoints, plus gate negative probes",
    )
    ap.add_argument(
        "--cpu-smoke-out",
        default=str(_REPO_ROOT / "eval_results" / "issue_2254" / FOLLOWUP_LABEL / "smoke"),
        help="evidence dir for --cpu-smoke summaries",
    )
    ap.add_argument(
        "--rig-health",
        action="store_true",
        help=(
            "POD-SIDE advisory probe (plan §4 smoke (d)): sycophancy × ctxext × (mid,+4) "
            "× 5q × 2 draws, judged at the production instrument; live API spend; "
            "advisory only — never a verdict input"
        ),
    )
    return ap


def _apply_smoke(args) -> None:
    """Tiny-real slice (plan §4 smoke): counts only — the phase code paths,
    gates, and dispatcher shape are identical; scratch out-root + smoke/ HF
    sub-prefix so smoke OUTPUTS never overwrite canonical ones; reused INPUTS
    stay canonical (module constants + canonical parent-vec loader)."""
    args.layers = list(PARITY_LAYERS)  # gate (ii) still spans its 4 registered cells
    args.q_steer = 2
    args.draws = 2
    if args.out_root == "eval_results/issue_2254":
        args.out_root = "/tmp/issue-2254-ladder-smoke"
    if args.fig_dir is None:
        args.fig_dir = str(Path(args.out_root) / "figures")
    i2254._SMOKE_UPLOAD_SUBPREFIX = True


def _dry_run_phase(args, phase: str) -> None:
    """Enumerate the phase grid + RESOLVE its deferred imports (no GPU/HF/
    model): a missing symbol / signature drift in a pod-only branch must fail
    HERE, not after the expensive phases (#606/#823/#1332)."""
    if phase == "directions":
        from concurrent.futures import ProcessPoolExecutor  # noqa: F401

        from threadpoolctl import threadpool_limits  # noqa: F401

        assert callable(i2254.map_svd) and callable(i2254.preimage_w)
        assert callable(i2254.destandardized_direction) and callable(i2254.kstar_from_fit)
        assert callable(i2254._save_direction) and callable(i2254._ensure_direction_vec)
        i2254._breadcrumb(SENTINEL_DIRECTIONS, dry_run=1, layers=len(args.layers))
    elif phase == "steer":
        from explore_persona_space.experiments.issue1415.steering import (  # noqa: F401
            DeltaHook,
            generate_batch,
        )
        from explore_persona_space.experiments.issue2254.hooks import (  # noqa: F401
            multi_layer_delta_hooks,
        )
        import scripts.issue2220_readwrite as rw2220

        assert callable(rw2220._pack_tree_to_jsonl_shards)
        assert callable(fk._assert_hub_headroom_for_steer)
        cells = registered_cells(args)
        i2254._breadcrumb(SENTINEL_STEER, dry_run=1, cells=len(cells))
    elif phase == "judge":
        from explore_persona_space.experiments.issue_1739.judging import (  # noqa: F401
            judge_items_graded,
            judge_tallies,
            load_trait_rubric,
            rollout_item_id,
        )
        from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: F401

        assert callable(fk._judge_graded_with_refusal_reissue)
        assert callable(fk._judge_instrument_fp)
        cells = registered_cells(args)
        i2254._breadcrumb("ladder-judge", dry_run=1, cells=len(cells))
    elif phase == "reduce":
        _ensure_reduce_git_inputs()
        fixture = assert_parent_reference_margin()
        for b in ROUND_BEHAVIORS:
            load_parent_floor(b)
            load_parent_band(b)
        i2254._breadcrumb("ladder-reduce", dry_run=1, fixture=fixture["verdict"])
    elif phase == "figures":
        import scripts.issue2254_ladder_figures as ladder_figs

        assert callable(ladder_figs.render_all)
        i2254._breadcrumb(SENTINEL_FIGURES, dry_run=1, required=len(ladder_figs.REQUIRED_FIGURES))
    else:  # pragma: no cover — main() validates phase names first
        raise SystemExit(f"unknown phase {phase!r}")


def run_phases(args, phases: list[str]) -> None:
    """Sequential phase dispatch: a HALT/raise in any phase stops the chain
    (the steer phase is never entered after a directions HALT — plan §7(a);
    pinned by the CPU test's spy)."""
    for p in phases:
        PHASES[p](args)


def main() -> None:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    if args.cpu_smoke:
        run_cpu_smoke(args)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    if args.rig_health:
        run_rig_health(args)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    if not args.phases:
        raise SystemExit(
            "--phases is required (comma-separated: directions,steer,judge,reduce,figures) "
            "or --import-check / --cpu-smoke / --rig-health"
        )
    phases = [p.strip() for p in args.phases.split(",") if p.strip()]
    unknown = [p for p in phases if p not in PHASES]
    if unknown:
        raise SystemExit(f"unknown phase(s) {unknown}; choices: {sorted(PHASES)}")
    if args.smoke:
        _apply_smoke(args)
    if args.dry_run:
        for p in phases:
            _dry_run_phase(args, p)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    run_phases(args, phases)
    # Explicit hard-exit after flush: this driver imports torch/transformers/HF
    # in its phases, so a finalize-time teardown race can rewrite the rc
    # (gotchas.md). Outputs are rename-atomic and uploaded before here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
