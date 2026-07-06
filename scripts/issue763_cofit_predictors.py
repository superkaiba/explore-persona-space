#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, λ, γ, Δ, ×, ≥) in scientific docstrings + log messages.
"""Issue #763 `neutral-contrast-and-cofit` Phase C: the matched-protocol co-fit.

8 predictors × 5 behaviors on the FROZEN graded E0 under ONE uniform protocol
(plan §4.2): shared LOCO folds (seed 763), within-fold rank target (UNWEIGHTED),
fixed PCA d=10 for subspace methods, per-method layer-max with SELECTION-
SYMMETRIC nulls (every draw re-runs the full per-fold refit across all 28
layers and takes the same layer-max; per-draw × per-layer matrices persisted to
``cofit_null_matrices/`` — #778 rule), K=100 random-direction floor, context
bootstrap B=2000, the frozen probe-aligned √r_yy ceilings, and the
``ridge_parent_check`` positive control. The pv_rA / pv_rC / pv_neutral
direction variants stay UNCOLLAPSED in every key (plan §5 label discipline).

Phase C continued (§4.3): the nonlinear-benefit block — kernel-vs-linear paired
sign-flip (B=10,000, rank scale) + the #742 plan-v9 "Option A" LEACE→dCor/HSIC
(single full-sample PCA-10 basis + single LEACE eraser, FULL pipeline refit per
permutation draw, B_perm=1000, three-part selectivity verdict) + the synthetic
power pre-check run FIRST through the same pipeline + the d=20 sensitivity leg
(exploratory) + the sign-flip minimal-detectable-|Δρ| simulation bound.

Fit-start step 0 writes ``inputs_manifest.json`` (sha256 + bytes + source for
every staged reused input) so the frozen-input claim is spot-checkable
(acceptance criterion 7: the E0 JSONs are read byte-untouched).

VECTORIZED (``analysis.issue_763_cofit``): observed + shuffle + control
batteries share ONE concatenated (1 + 2·n_perms)-draw batch per behavior so
every label-independent per-(layer, fold) cache (PCA basis, standardization,
kernel eigendecompositions, diff-means cross-Grams) is built ONCE and reused;
the ``assert_cofit_matches_reference`` + ``assert_option_a_contract`` exactness
gates run at start-up before any behavior is fit.

``--smoke``: 1 behavior × 8 contexts × tiny perms on the REAL local v0/E0/rb
inputs, with SURROGATE ``pv_directions_v2`` / ``c0`` artifacts written through
the production writer schema and read back through the production loaders (the
writer/reader round-trip is exercised at production dims; the real Phase-A/B
artifacts replace them at run time).

Usage::

    uv run python scripts/issue763_cofit_predictors.py                # full run
    uv run python scripts/issue763_cofit_predictors.py --smoke
    EPM_FIT_DEVICE=cuda uv run python scripts/issue763_cofit_predictors.py
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# #847: shared-VM thread caps must bind BEFORE torch/numpy freeze their pools at import.
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue594_common import load_battery  # noqa: E402
from issue658_fit_predictors import (  # noqa: E402
    DEVICE,
    _cluster_bootstrap_rho,
    _resolve_device,
)
from issue763_common import (  # noqa: E402
    BEHAVIORS,
    C0_SHARD_DIR,
    COFIT_DIR,
    COFIT_NULL_MATRIX_DIR,
    EVAL_RESULTS_DIR,
    HF_ANALYSIS_TENSORS_PREFIX,
    HF_DATA_REPO,
    PV_DIRECTIONS_V2_DIR,
    SEED,
    assert_pool_floor,
    assert_production_direction_shape,
    dump_json,
    ensure_smoke_scope,
    load_json,
    reproducibility_metadata,
    sha256_file,
    smoke_scoped,
)
from issue763_fit_predictors import _e0_vectors, _load_v0  # noqa: E402

from explore_persona_space.analysis.issue_763_cofit import (  # noqa: E402
    COFIT_LAMBDAS,
    COFIT_PCA_DIM,
    COFIT_RBF_MULTIPLIERS,
    LayerCache,
    assert_cofit_matches_reference,
    batched_spearman,
    diffmeans_loco_preds,
    direction_loco_preds,
    fold_rank_targets,
    kernel_loco_preds,
    random_unit_directions,
    rank01,
)
from explore_persona_space.analysis.issue_763_nonlinear import (  # noqa: E402
    assert_option_a_contract,
    dcor_power_check,
    paired_signflip_test,
    run_option_a_cell,
    signflip_min_detectable_delta_rho,
)

logger = logging.getLogger("issue763_cofit")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FIT_DEVICE = _resolve_device(os.environ.get("EPM_FIT_DEVICE", DEVICE))

N_PERMS = 1000
N_BOOT = 2000
K_RANDOM = 100
SIGNFLIP_B = 10_000
NONLINEAR_B_PERM = 1000
POWER_TRIALS = 200
RBF_LABELS = tuple(f"rbf_c{c}" for c in COFIT_RBF_MULTIPLIERS)

# The 8 co-fit methods (plan §5 config slugs; pv_* stay UNCOLLAPSED everywhere).
METHODS: tuple[str, ...] = (
    "cofit_ridge",
    "cofit_krr",
    "pv_rA",
    "pv_rC",
    "pv_neutral",
    "diffmeans_crude",
    "rand_dir",
    "cC_ridge",
)
# Plain-English names carried into the results JSON (label discipline, plan §5).
METHOD_LABELS: dict[str, str] = {
    "cofit_ridge": "learned ridge (answer-side)",
    "cofit_krr": "kernel ridge (nonlinear)",
    "pv_rA": "opposite-contrast direction, instruction-present read",
    "pv_rC": "opposite-contrast direction, stripped read",
    "pv_neutral": "neutral-contrast direction (new arm)",
    "diffmeans_crude": "crude supervised diff-means",
    "rand_dir": "random-direction floor (K seeded unit directions)",
    "cC_ridge": "prompt-side context ridge",
}

REANCHOR_DIR = EVAL_RESULTS_DIR / "deception-rubric-reanchor"
PARENT_RESULTS_PATH = REANCHOR_DIR / "matched_predictor_results.json"
E0_PARENT_PATH = EVAL_RESULTS_DIR / "E0_matched_by_behavior.json"
E0_DECEPTION_V2_PATH = REANCHOR_DIR / "E0_deception_v2.json"
# smoke_scoped: the dispatcher smoke's Phase-1 prereq WRITES a tiny mock rb
# shard here — under the scope env it lands in smoke_scope/, never clobbering
# the real committed shards (review r1 C1(iii)).
PV_SHARD_DIR = smoke_scoped(EVAL_RESULTS_DIR / "pv_shards")

# Ridge positive-control bands (plan §4.2: ±0.12 soft, ±0.25 hard fail-loud).
PARENT_CHECK_SOFT = 0.12
PARENT_CHECK_HARD = 0.25


# ── step 0: inputs manifest (sha256-pinned frozen inputs) ─────────────────────


def _stage_round_inputs(behaviors: list[str]) -> None:
    """Stage EVERY ``write_inputs_manifest`` frozen input from HF (fail-loud).

    v0 shards stage EAGERLY here via the parent's ``_stage_v0_shards_from_hf``
    — ``write_inputs_manifest`` (fit-start step 0) sha256-pins them BEFORE the
    first lazy ``_load_v0`` read at battery time, so lazy-only staging crashes
    any fresh lane with no local worktree copy at manifest step 0 (r3
    crash-fix; ``_load_v0``'s lazy staging stays as a no-op backstop). E0 + rb
    shards stage via the parent's ``_stage_fit_inputs_from_hf``; the round's
    own pv_directions_v2 + c0 shards (produced by Phases A/B) stage here from
    the round's HF prefixes. The remaining ``write_inputs_manifest``
    frozen-input pins (pv_rollouts + pv_judge_v2 — gitignored ``data/`` paths
    a fresh eval-lane clone lacks) stage here too (review r1 C2), then get
    validated against the ABSOLUTE plan-registered pool size so a stale mock
    at a canonical path can never be sha256-pinned into
    ``inputs_manifest.json`` as a "frozen input" (review r1 C1(iii)). The
    manifest's two non-staged inputs (E0_deception_v2 + the reanchor parent
    record) are git-tracked, so a fresh clone carries them; the coverage
    invariant is pinned by
    ``tests/test_issue763_cofit_phase_boundary.py::test_every_manifest_frozen_input_is_staged_or_git_tracked``.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError
    from issue763_extract_pv_rb import (
        PV_JUDGE_V2_DIR,
        PV_POOL_EXPECTED,
        PV_ROLLOUT_DIR,
        _read_rollouts_jsonl,
        _stage_from_hf,
    )
    from issue763_fit_predictors import _stage_fit_inputs_from_hf, _stage_v0_shards_from_hf

    _stage_fit_inputs_from_hf(behaviors, stage_e0=True)
    _stage_v0_shards_from_hf(behaviors)
    _stage_from_hf("pv_rollouts", PV_ROLLOUT_DIR, behaviors, suffix="jsonl")
    _stage_from_hf("pv_judge_v2", PV_JUDGE_V2_DIR, behaviors, suffix="json")
    for b in behaviors:
        rows = _read_rollouts_jsonl(PV_ROLLOUT_DIR / f"{b}.jsonl")
        assert_pool_floor(len(rows), PV_POOL_EXPECTED, f"{b}: pv_rollouts (manifest input)")
        flags = load_json(PV_JUDGE_V2_DIR / f"{b}.json")["keep_flags"]
        if len(flags) != len(rows):
            raise RuntimeError(
                f"{b}: pv_judge_v2 keep_flags ({len(flags)}) != pv_rollouts rows "
                f"({len(rows)}) — the keep-flags were judged on a DIFFERENT rollout "
                "pool (stale smoke residue?); purge + re-stage from HF"
            )

    def _fetch(local: Path, filename: str, what: str) -> None:
        if local.exists():
            return
        try:
            src = hf_hub_download(repo_id=HF_DATA_REPO, repo_type="dataset", filename=filename)
        except EntryNotFoundError as e:
            raise FileNotFoundError(
                f"{what} is neither local ({local}) nor on HF ({HF_DATA_REPO}/{filename}) — "
                "run the Phase-A capture + Phase-B assembly (and their uploads) first"
            ) from e
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(Path(src).read_bytes())

    for b in behaviors:
        _fetch(
            PV_DIRECTIONS_V2_DIR / f"{b}.pt",
            f"{HF_ANALYSIS_TENSORS_PREFIX}/pv_directions_v2/{b}.pt",
            f"pv_directions_v2/{b}.pt",
        )
        _fetch(
            C0_SHARD_DIR / f"c0_{b}.pt",
            f"{HF_ANALYSIS_TENSORS_PREFIX}/c0_shards/c0_{b}.pt",
            f"c0_shards/c0_{b}.pt",
        )


def write_inputs_manifest(behaviors: list[str], *, smoke: bool) -> dict:
    """Fit-start step 0: sha256 + bytes + source for every staged reused input.

    Committed with the results + mirrored to HF so the frozen-input claim is
    spot-checkable (plan §4.2 step 0; acceptance criterion 7 — the E0 JSONs are
    hashed exactly as read, byte-untouched).
    """
    from issue763_extract_pv_rb import PV_JUDGE_V2_DIR, PV_ROLLOUT_DIR

    entries: list[dict] = []

    def _add(path: Path, source: str) -> None:
        if not path.exists():
            if smoke:
                return  # smoke tolerates absent optional inputs (tiny slice)
            raise FileNotFoundError(f"inputs_manifest: frozen input missing: {path}")
        entries.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
                "source": source,
            }
        )

    hfp = f"{HF_DATA_REPO}/{HF_ANALYSIS_TENSORS_PREFIX}"
    for b in behaviors:
        _add(PV_ROLLOUT_DIR / f"{b}.jsonl", f"{hfp}/pv_rollouts/{b}.jsonl")
        _add(PV_JUDGE_V2_DIR / f"{b}.json", f"{hfp}/pv_judge_v2/{b}.json")
        _add(EVAL_RESULTS_DIR / "v0_shards" / f"v0_{b}.pt", f"{hfp}/v0_shards/v0_{b}.pt")
        _add(PV_SHARD_DIR / f"rb_{b}.pt", f"{hfp}/pv_shards/rb_{b}.pt")
        _add(PV_DIRECTIONS_V2_DIR / f"{b}.pt", "round Phase A/B (this run)")
        _add(C0_SHARD_DIR / f"c0_{b}.pt", "round Phase A (this run)")
    _add(E0_PARENT_PATH, f"{hfp}/E0_matched_by_behavior.json")
    _add(E0_DECEPTION_V2_PATH, "git issue-763 (deception-rubric-reanchor, FROZEN)")
    _add(PARENT_RESULTS_PATH, "git issue-763 (reanchor record: parent ρ + ceilings)")
    manifest = {
        "round": "neutral-contrast-and-cofit",
        "inputs": entries,
        "metadata": reproducibility_metadata({"phase": "cofit_inputs_manifest"}),
    }
    dump_json(manifest, COFIT_DIR / "inputs_manifest.json")
    logger.info("[manifest] pinned %d frozen inputs -> inputs_manifest.json", len(entries))
    return manifest


# ── input loaders ─────────────────────────────────────────────────────────────


def _load_graded_e0(behavior: str) -> dict:
    """The FROZEN graded target record: v2 rubric for deception, v1 for the rest."""
    if behavior == "deception":
        return load_json(E0_DECEPTION_V2_PATH)
    return load_json(E0_PARENT_PATH)


def load_directions_v2(behavior: str, dirs_dir: Path) -> dict:
    """Load the round's direction blob; assert the production writer schema."""
    blob = torch.load(dirs_dir / f"{behavior}.pt", weights_only=False)
    for key in ("r_C", "r_neutral", "keep_floor_branch"):
        assert key in blob, f"pv_directions_v2/{behavior}.pt missing key {key!r}"
    return blob


def _load_c0(behavior: str, c0_dir: Path) -> tuple[np.ndarray, list[str]]:
    blob = torch.load(c0_dir / f"c0_{behavior}.pt", weights_only=False)
    return blob["tensor"].float().numpy(), blob["context_ids"]


def _parent_record(behavior: str) -> dict:
    rec = load_json(PARENT_RESULTS_PATH)
    return rec["by_behavior"][behavior] if "by_behavior" in rec else rec[behavior]


# ── the per-behavior battery (observed + shuffle + control in ONE batch) ─────


def run_behavior_battery(
    v0k: np.ndarray,
    c0k: np.ndarray | None,
    y: np.ndarray,
    *,
    directions: dict[str, np.ndarray | None],
    n_perms: int,
    seed: int,
    k_random: int,
    dim: int,
    device: str,
    methods: tuple[str, ...] = METHODS,
) -> dict:
    """All methods × all layers × (observed + shuffle + control) in one pass.

    The three batteries are CONCATENATED into one (1 + 2·n_perms, n) label
    batch so the per-(layer, fold) label-independent caches are built once.
    Returns per method: ``per_layer_rho_obs`` (L,), ``rho_shuffle`` (L, P),
    ``rho_control`` (L, P), ``preds_obs`` (L, n); plus the random-direction
    observed matrix ``rand_per_dir_layer_rho`` (K, L).
    """
    n, n_layers, hidden = v0k.shape
    rng_shuffle = np.random.default_rng(seed)
    rng_control = np.random.default_rng(seed + 7)
    y_shuffle = np.stack([y[rng_shuffle.permutation(n)] for _ in range(n_perms)])
    y_control = np.stack([y[rng_control.permutation(n)] for _ in range(n_perms)])
    y_all = np.concatenate([y[None, :], y_shuffle, y_control], axis=0)  # (1+2P, n)
    R = fold_rank_targets(y_all)
    R_obs = R[:1]

    rand_dirs = random_unit_directions(k_random, hidden, seed=seed) if k_random else None

    out: dict = {m: {} for m in methods}
    for m in methods:
        out[m]["per_layer_rho_obs"] = np.full(n_layers, np.nan)
        out[m]["rho_shuffle"] = np.full((n_layers, n_perms), np.nan)
        out[m]["rho_control"] = np.full((n_layers, n_perms), np.nan)
        out[m]["preds_obs"] = np.full((n_layers, n), np.nan)
    rand_rho = np.full((k_random, n_layers), np.nan) if k_random else None

    def _record(m: str, ell: int, preds: np.ndarray) -> None:
        rho = batched_spearman(preds, y_all)
        out[m]["per_layer_rho_obs"][ell] = rho[0]
        out[m]["rho_shuffle"][ell] = rho[1 : 1 + n_perms]
        out[m]["rho_control"][ell] = rho[1 + n_perms :]
        out[m]["preds_obs"][ell] = preds[0]

    for ell in range(n_layers):
        x = v0k[:, ell, :].astype(np.float64)
        needs_v0_cache = any(m in methods for m in ("cofit_ridge", "cofit_krr", "diffmeans_crude"))
        cache = (
            LayerCache.build(x, dim=dim, rbf_multipliers=COFIT_RBF_MULTIPLIERS, device=device)
            if needs_v0_cache
            else None
        )
        if "cofit_ridge" in methods:
            _record("cofit_ridge", ell, kernel_loco_preds(cache, R, kernel_labels=("linear",)))
        if "cofit_krr" in methods:
            _record("cofit_krr", ell, kernel_loco_preds(cache, R, kernel_labels=RBF_LABELS))
        if "diffmeans_crude" in methods:
            _record("diffmeans_crude", ell, diffmeans_loco_preds(cache, R, y_all))
        for m, key in (("pv_rA", "r_A"), ("pv_rC", "r_C"), ("pv_neutral", "r_neutral")):
            if m not in methods:
                continue
            r = directions.get(key)
            if r is None:
                continue  # unbuildable arm — the record carries the branch
            _record(m, ell, direction_loco_preds(x @ r[ell], R))
        if "cC_ridge" in methods and c0k is not None:
            xc = c0k[:, ell, :].astype(np.float64)
            cache_c = LayerCache.build(xc, dim=dim, rbf_multipliers=(), device=device)
            _record("cC_ridge", ell, kernel_loco_preds(cache_c, R, kernel_labels=("linear",)))
        if rand_dirs is not None:
            s_rand = x @ rand_dirs.T  # (n, K)
            for k in range(k_random):
                preds_k = direction_loco_preds(s_rand[:, k], R_obs)
                rand_rho[k, ell] = batched_spearman(preds_k, y[None, :])[0]
        if ell % 7 == 0:
            logger.info("[battery] layer %d/%d done", ell + 1, n_layers)
    out["rand_per_dir_layer_rho"] = rand_rho
    return out


def _layer_max_stats(rho_matrix: np.ndarray, obs_max: float | None) -> dict:
    """Layer-max-per-draw null stats (+1-corrected right-tail p; all-NaN dropped)."""
    valid = ~np.all(np.isnan(rho_matrix), axis=0)  # (P,)
    if not valid.any() or obs_max is None:
        return {"p_value": None, "null_p95": None, "n_draws": 0}
    null = np.nanmax(rho_matrix[:, valid], axis=0)
    p = float((np.sum(null >= obs_max) + 1) / (len(null) + 1))
    return {
        "p_value": p,
        "null_p95": float(np.percentile(null, 95)),
        "n_draws": len(null),
    }


def _obs_layer_max(per_layer: np.ndarray) -> tuple[int | None, float | None]:
    if np.all(np.isnan(per_layer)):
        return None, None
    ell = int(np.nanargmax(per_layer))
    return ell, float(per_layer[ell])


def _pooled_rho_excluding(preds: np.ndarray, y: np.ndarray, mask_keep: np.ndarray) -> float | None:
    """Pooled Spearman recomputed on a context subset (leave-default-family-out)."""
    if mask_keep.sum() < 4:
        return None
    r = batched_spearman(preds[None, mask_keep], y[None, mask_keep])[0]
    return None if np.isnan(r) else float(r)


# ── per-behavior driver ───────────────────────────────────────────────────────


def _behavior_config_key(behavior: str, args) -> dict:
    """The per-behavior resume/checkpoint key — regime knobs + INPUT identity.

    The input sha256s (direction blob + c0 shard) make the resume predicate
    input-AWARE: a Group-C re-dispatch after a Phase-A/B fix must refit, never
    silently reuse fits keyed only on {n_perms, n_boot, k_random, smoke}
    (review r1 Minor; the #722-r3 resume-regime-key class).
    """
    return {
        "n_perms": args.n_perms,
        "n_boot": args.n_boot,
        "k_random": args.k_random,
        "smoke": bool(args.smoke),
        "directions_sha256": sha256_file(Path(args.dirs_dir) / f"{behavior}.pt"),
        "c0_sha256": sha256_file(Path(args.c0_dir) / f"c0_{behavior}.pt"),
    }


def fit_behavior_cofit(behavior: str, args, fam_by_ctx: dict[str, str]) -> dict:
    """One behavior's full co-fit record (checkpointed to fit_by_behavior/)."""
    v0, ctx_ids = _load_v0(behavior)
    e0 = _load_graded_e0(behavior)
    graded, rates, _njudged, _bw, _ppg, _ppb, kept = _e0_vectors(e0, behavior, ctx_ids)
    keep_pos = [ctx_ids.index(c) for c in kept]
    v0k = v0[keep_pos]
    if args.smoke:
        v0k, kept, graded, rates = v0k[:8], kept[:8], graded[:8], rates[:8]
    n = len(kept)
    assert n >= 4, f"{behavior}: only {n} contexts with a graded_mean"

    c0, c0_ids = _load_c0(behavior, Path(args.c0_dir))
    c0_index = {c: i for i, c in enumerate(c0_ids)}
    missing_c0 = [c for c in kept if c not in c0_index]
    assert not missing_c0, f"{behavior}: c0 shard missing contexts {missing_c0[:4]}"
    c0k = c0[[c0_index[c] for c in kept]]

    dir_blob = load_directions_v2(behavior, Path(args.dirs_dir))
    if args.smoke and "r_A_surrogate" in dir_blob:
        # smoke: the unified dispatcher's prereq overwrites the local rb shard
        # with tiny-model dims; the v0-dim-consistent surrogate carries r_A.
        rb = dir_blob["r_A_surrogate"].float().numpy()
    else:
        rb = (
            torch.load(PV_SHARD_DIR / f"rb_{behavior}.pt", weights_only=False)["r_b"]
            .float()
            .numpy()
        )
    directions = {
        "r_A": rb.astype(np.float64),
        "r_C": dir_blob["r_C"].float().numpy().astype(np.float64),
        "r_neutral": (
            None
            if dir_blob["r_neutral"] is None
            else dir_blob["r_neutral"].float().numpy().astype(np.float64)
        ),
    }
    if not args.smoke:
        # ABSOLUTE production-dim validation (review r1 C1(iii)): a staged
        # direction with tiny-smoke-model dims must fail loud BEFORE fitting.
        for name, arr in directions.items():
            if arr is not None:
                assert_production_direction_shape(arr.shape, f"{behavior}: direction {name}")
    y = np.asarray(graded, dtype=np.float64)

    bat = run_behavior_battery(
        v0k,
        c0k,
        y,
        directions=directions,
        n_perms=args.n_perms,
        seed=SEED,
        k_random=args.k_random,
        dim=COFIT_PCA_DIM,
        device=FIT_DEVICE,
    )

    parent = _parent_record(behavior)
    sqrt_r_yy = parent.get("sqrt_r_yy_graded")
    parent_layer = parent.get("chosen_layer")
    parent_rho = parent.get("rho_graded_ridge")

    COFIT_NULL_MATRIX_DIR.mkdir(parents=True, exist_ok=True)
    default_mask = np.array([fam_by_ctx.get(c) != "default" for c in kept])

    rec: dict = {
        "behavior": behavior,
        "n_contexts": n,
        "kept_context_ids": kept,
        # per-context graded target + family — the pred-vs-actual scatter's axes
        # (review r1 Minor: the low-level data plot needs the E0 axis + family
        # coloring; persisting both keeps the plot script input-self-contained).
        "graded_by_context": {c: float(v) for c, v in zip(kept, graded, strict=True)},
        "family_by_context": {c: fam_by_ctx.get(c) for c in kept},
        "sqrt_r_yy_graded": sqrt_r_yy,
        "protocol": {
            "target": "graded E0 rank-transformed within training fold (UNWEIGHTED)",
            "folds": "LOCO, shared list, seed 763",
            "pca_dim": COFIT_PCA_DIM,
            "lambdas": list(COFIT_LAMBDAS),
            "rbf_multipliers": list(COFIT_RBF_MULTIPLIERS),
            "n_perms": args.n_perms,
            "n_boot": args.n_boot,
            "k_random": args.k_random,
        },
        "methods": {},
    }

    for m in METHODS:
        label = METHOD_LABELS[m]
        if m == "rand_dir":
            rand = bat["rand_per_dir_layer_rho"]  # (K, L)
            torch.save(
                {"matrix": torch.from_numpy(rand).to(torch.float32), "behavior": behavior},
                COFIT_NULL_MATRIX_DIR / f"{behavior}_rand_dir_observed.pt",
            )
            layer_max = np.nanmax(rand, axis=1)  # (K,)
            band = (
                [float(np.percentile(layer_max, 2.5)), float(np.percentile(layer_max, 97.5))]
                if len(layer_max)
                else None
            )
            rec["methods"][m] = {
                "label": label,
                "rho": float(np.median(layer_max)),  # band median (NOT a fit read)
                "ci95": band,  # the 2.5–97.5 selection-floor band, NOT a bootstrap CI
                "ci95_is_selection_band": True,
                "chosen_layer": None,
                "null_p": None,
                "per_layer_rho": [
                    None if np.isnan(v) else float(v) for v in np.nanmedian(rand, axis=0)
                ],
                "band_p97_5": band[1] if band else None,
                "k_random": args.k_random,
                "note": "selection floor: per-dir layer-max distribution over K seeded dirs",
            }
            continue
        per_layer = bat[m]["per_layer_rho_obs"]
        if m == "pv_neutral" and directions["r_neutral"] is None:
            rec["methods"][m] = {
                "label": label,
                "rho": None,
                "ci95": None,
                "chosen_layer": None,
                "null_p": None,
                "per_layer_rho": None,
                "unbuildable": True,
                "keep_floor_branch": dir_blob["keep_floor_branch"],
                "note": "r_neutral UNBUILDABLE (< hard keep-floor) — a finding, plan §4.1.2",
            }
            continue
        ell, obs_rho = _obs_layer_max(per_layer)
        shuffle_stats = _layer_max_stats(bat[m]["rho_shuffle"], obs_rho)
        control_stats = _layer_max_stats(bat[m]["rho_control"], obs_rho)
        control_pass = (
            obs_rho is not None
            and control_stats["null_p95"] is not None
            and obs_rho > control_stats["null_p95"]
        )
        for battery_name, matrix in (
            ("shuffle", bat[m]["rho_shuffle"]),
            ("control", bat[m]["rho_control"]),
        ):
            torch.save(
                {
                    "matrix": torch.from_numpy(matrix.T).to(torch.float32),  # (P, L)
                    "behavior": behavior,
                    "method": m,
                    "battery": battery_name,
                    "seed": SEED if battery_name == "shuffle" else SEED + 7,
                },
                COFIT_NULL_MATRIX_DIR / f"{behavior}_{m}_{battery_name}.pt",
            )
        preds_chosen = bat[m]["preds_obs"][ell] if ell is not None else None
        boot = (
            _cluster_bootstrap_rho(preds_chosen, y, n_boot=args.n_boot, seed=SEED)
            if preds_chosen is not None
            else None
        )
        entry = {
            "label": label,
            "rho": obs_rho,
            "ci95": boot["ci95"] if boot else None,
            "chosen_layer": ell,
            "null_p": shuffle_stats["p_value"],
            "shuffle_null_p95": shuffle_stats["null_p95"],
            "control_task_pass": bool(control_pass),
            "control_p95": control_stats["null_p95"],
            "per_layer_rho": [None if np.isnan(v) else float(v) for v in per_layer],
            "preds_chosen_layer": (
                None
                if preds_chosen is None
                else {c: float(p) for c, p in zip(kept, preds_chosen, strict=True)}
            ),
            "rho_leave_default_family_out": (
                None
                if preds_chosen is None
                else _pooled_rho_excluding(preds_chosen, y, default_mask)
            ),
        }
        if m == "pv_neutral":
            entry["keep_floor_branch"] = dir_blob["keep_floor_branch"]
            entry["pv_thin_sample"] = dir_blob["keep_floor_branch"] == "thin"
        rec["methods"][m] = entry

    # ── ridge positive control (plan §4.2; kill criterion (a)) ──
    ridge_layers = rec["methods"]["cofit_ridge"]["per_layer_rho"]
    harness_at_parent = (
        ridge_layers[parent_layer]
        if parent_layer is not None and ridge_layers and parent_layer < len(ridge_layers)
        else None
    )
    delta = (
        None
        if harness_at_parent is None or parent_rho is None
        else float(harness_at_parent - parent_rho)
    )
    rec["ridge_parent_check"] = {
        "parent_chosen_layer": parent_layer,
        "parent_rho_graded_ridge": parent_rho,
        "harness_rho_at_parent_layer": harness_at_parent,
        "delta": delta,
        "soft_band": PARENT_CHECK_SOFT,
        "hard_band": PARENT_CHECK_HARD,
        "within_soft": delta is not None and abs(delta) <= PARENT_CHECK_SOFT,
        "within_hard": delta is not None and abs(delta) <= PARENT_CHECK_HARD,
    }
    if delta is not None and abs(delta) > PARENT_CHECK_HARD and not args.smoke:
        raise RuntimeError(
            f"{behavior}: ridge_parent_check FAILED hard band — harness ridge at parent "
            f"layer {parent_layer} = {harness_at_parent:.3f} vs parent {parent_rho:.3f} "
            f"(|Δ|={abs(delta):.3f} > {PARENT_CHECK_HARD}) — debug the harness before "
            "interpreting cross-method gaps (plan §7 kill criterion a)"
        )
    if delta is not None and abs(delta) > PARENT_CHECK_SOFT:
        logger.warning(
            "[parent-check] %s: |Δ|=%.3f outside the ±%.2f soft band (investigate)",
            behavior,
            abs(delta),
            PARENT_CHECK_SOFT,
        )

    # ── binary-companion ridge column (continuity with the parent's binary reads) ──
    bin_mask = ~np.isnan(rates)
    if bin_mask.sum() >= 4:
        y_bin = rates[bin_mask].astype(np.float64)
        bat_bin = run_behavior_battery(
            v0k[bin_mask],
            None,
            y_bin,
            directions={"r_A": None, "r_C": None, "r_neutral": None},
            n_perms=args.n_perms,
            seed=SEED,
            k_random=0,
            dim=COFIT_PCA_DIM,
            device=FIT_DEVICE,
            methods=("cofit_ridge",),
        )
        ell_b, rho_b = _obs_layer_max(bat_bin["cofit_ridge"]["per_layer_rho_obs"])
        stats_b = _layer_max_stats(bat_bin["cofit_ridge"]["rho_shuffle"], rho_b)
        rec["binary_companion_ridge"] = {
            "rho": rho_b,
            "chosen_layer": ell_b,
            "null_p": stats_b["p_value"],
            "n_contexts": int(bin_mask.sum()),
            "target": "binary expressed-rate, rank-transformed (ridge column only)",
        }
    else:
        rec["binary_companion_ridge"] = None

    rec["config"] = _behavior_config_key(behavior, args)
    rec["metadata"] = reproducibility_metadata({"phase": "cofit_fit", "behavior": behavior})
    return rec


# ── nonlinear-benefit block (plan §4.3) ───────────────────────────────────────


def run_nonlinear_block(behaviors: list[str], records: dict[str, dict], args) -> dict:
    """§4.3: power pre-check FIRST, then per-behavior sign-flip + Option-A cells."""
    rng = np.random.default_rng(SEED)
    n_ctx = min(rec["n_contexts"] for rec in records.values())
    n_perm = args.nonlinear_b_perm
    trials = args.power_trials

    # (3) synthetic power pre-check THROUGH the same pipeline, run FIRST; the
    # (d, n) config is behavior-independent so it is computed once per d.
    power: dict[str, float] = {}
    for d in (10, 20):
        power[f"d{d}"] = dcor_power_check(
            d_eff=d, n=n_ctx, n_perm=n_perm, effect=0.10, n_trials=trials, rng=rng
        )
        logger.info("[nonlinear] power(d=%d, n=%d) = %.3f", d, n_ctx, power[f"d{d}"])
    min_detect = signflip_min_detectable_delta_rho(
        n=n_ctx,
        n_trials=trials,
        n_flips=max(200, args.signflip_b // 10),
        rng=rng,
    )

    cells: dict[str, dict] = {}
    for behavior in behaviors:
        rec = records[behavior]
        kept = rec["kept_context_ids"]
        v0, ctx_ids = _load_v0(behavior)
        keep_pos = [ctx_ids.index(c) for c in kept]
        v0k = v0[keep_pos]
        e0 = _load_graded_e0(behavior)
        graded, *_rest, _kept_all = _e0_vectors(e0, behavior, ctx_ids)
        y = np.asarray(graded[: len(kept)], dtype=np.float64)

        ridge = rec["methods"]["cofit_ridge"]
        krr = rec["methods"]["cofit_krr"]
        if (
            ridge.get("chosen_layer") is None
            or not ridge.get("preds_chosen_layer")
            or not krr.get("preds_chosen_layer")
        ):
            # degenerate slice (all-NaN ridge/KRR read) — record and move on,
            # never crash (review r1 unaddressed-case: pk was unguarded)
            cells[behavior] = {"skipped": "ridge or krr produced no valid layer read"}
            logger.warning("[nonlinear] %s: skipped (no valid ridge/krr layer)", behavior)
            continue
        # (1) kernel-vs-linear paired sign-flip on the RANK scale, both methods at
        # their OWN chosen layers; per-context error vs the full-sample rank01(y).
        y_rank = rank01(y)
        pl = np.array([ridge["preds_chosen_layer"][c] for c in kept])
        pk = np.array([krr["preds_chosen_layer"][c] for c in kept])
        if np.isnan(pl).any() or np.isnan(pk).any():
            cells[behavior] = {"skipped": "NaN in chosen-layer predictions (ridge or krr)"}
            logger.warning("[nonlinear] %s: skipped (NaN predictions)", behavior)
            continue
        err_lin = (pl - y_rank) ** 2
        err_krr = (pk - y_rank) ** 2
        signflip = paired_signflip_test(
            err_lin, err_krr, n_flips=args.signflip_b, rng=np.random.default_rng(SEED + 11)
        )
        paired_delta_rho = (
            None if ridge["rho"] is None or krr["rho"] is None else float(krr["rho"] - ridge["rho"])
        )

        # (2) Option-A LEACE→dCor/HSIC at the RIDGE's chosen layer; d=10 verdict +
        # d=20 exploratory sensitivity leg.
        ell = ridge["chosen_layer"]
        v0_layer = v0k[:, ell, :].astype(np.float64)
        cell_d10 = run_option_a_cell(
            v0_layer,
            y,
            behavior=behavior,
            layer=ell,
            d_eff=10,
            n_perm=n_perm,
            rng=np.random.default_rng(SEED + 13),
            realized_power=power["d10"],
        )
        cell_d20 = run_option_a_cell(
            v0_layer,
            y,
            behavior=behavior,
            layer=ell,
            d_eff=20,
            n_perm=n_perm,
            rng=np.random.default_rng(SEED + 17),
            realized_power=power["d20"],
        )
        cells[behavior] = {
            "signflip": signflip,
            "paired_delta_rho_krr_minus_ridge": paired_delta_rho,
            "option_a_d10": cell_d10,
            "option_a_d20_sensitivity": cell_d20,
            "h4_joint_falsification": bool(
                signflip["p_value"] < 0.05 and cell_d10["verdict"] == "nonlinear-yes"
            ),
        }
        logger.info(
            "[nonlinear] %s: signflip p=%.4f, d10 verdict=%s",
            behavior,
            signflip["p_value"],
            cell_d10["verdict"],
        )

    return {
        "round": "neutral-contrast-and-cofit",
        "protocol": (
            "Option A (#742 plan v9): single full-sample PCA basis + single LEACE "
            "eraser, FULL PCA->LEACE->dCor/HSIC pipeline refit per permutation draw; "
            "three-part selectivity verdict (alpha=0.05, delta_sel=0.10); paired "
            f"sign-flip B={args.signflip_b}; power pre-check through the same pipeline"
        ),
        "realized_power": power,
        "power_floor": 0.8,
        "signflip_min_detectable_delta_rho": min_detect,
        "n_perm": n_perm,
        "by_behavior": cells,
        "metadata": reproducibility_metadata({"phase": "cofit_nonlinear"}),
    }


# ── smoke surrogates (production writer schema, production loaders) ──────────


def _write_smoke_surrogates(behaviors: list[str], dirs_dir: Path, c0_dir: Path) -> None:
    """Write SURROGATE pv_directions_v2 + c0 shards for the --smoke slice.

    Derived from the REAL v0 shards (production key schema, v0-consistent
    (L, H) dims) so the co-fit exercises the exact consumption path; the real
    Phase-A/B artifacts replace them in the production run. The direction base
    is the REAL rb shard when its dims match v0 (a pristine worktree), else a
    seeded random (L, H) direction — the unified dispatcher smoke's Phase-1
    prereq OVERWRITES the local rb shard with a tiny-smoke-model one (L=2, H=8)
    while the v0 shards stay real, so rb dims cannot be assumed. The surrogate
    blob also carries ``r_A_surrogate`` so the smoke's pv_rA read is
    dim-consistent (production reads the real rb shard directly). Never written
    outside --smoke.
    """
    rng = np.random.default_rng(SEED)
    dirs_dir.mkdir(parents=True, exist_ok=True)
    c0_dir.mkdir(parents=True, exist_ok=True)
    for b in behaviors:
        v0, ctx_ids = _load_v0(b)
        n_layers, hidden = v0.shape[1], v0.shape[2]
        rb_path = PV_SHARD_DIR / f"rb_{b}.pt"
        base = None
        if rb_path.exists():
            rb = torch.load(rb_path, weights_only=False)["r_b"].float()
            if tuple(rb.shape) == (n_layers, hidden):
                base = rb
        if base is None:
            base = torch.from_numpy(rng.standard_normal((n_layers, hidden))).to(torch.float32)
        noise = torch.from_numpy(rng.standard_normal(base.shape)).to(torch.float32)
        scale = base.norm(dim=-1, keepdim=True) / max(float(noise.norm()), 1e-9)
        torch.save(
            {
                "behavior": b,
                "r_A_surrogate": base,
                "r_C": base + 0.05 * noise * scale,
                "r_neutral": base + 0.10 * noise * scale,
                "pos_C_mean": base,
                "neg_C_mean": torch.zeros_like(base),
                "neutral_mean": torch.zeros_like(base),
                "neutral_kept_n": 999,
                "keep_floor_branch": "normal",
                "read_context": "SMOKE_SURROGATE (v0-dim-consistent)",
            },
            dirs_dir / f"{b}.pt",
        )
        c0 = torch.from_numpy(v0 + 0.1 * rng.standard_normal(v0.shape)).to(torch.float32)
        torch.save(
            {"tensor": c0, "context_ids": ctx_ids, "behavior": b, "span": "prompt"},
            c0_dir / f"c0_{b}.pt",
        )


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #763 neutral-contrast-and-cofit Phase C.")
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n-perms", type=int, default=N_PERMS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--k-random", type=int, default=K_RANDOM)
    ap.add_argument("--signflip-b", type=int, default=SIGNFLIP_B)
    ap.add_argument("--nonlinear-b-perm", type=int, default=NONLINEAR_B_PERM)
    ap.add_argument("--power-trials", type=int, default=POWER_TRIALS)
    ap.add_argument("--dirs-dir", default=str(PV_DIRECTIONS_V2_DIR))
    ap.add_argument("--c0-dir", default=str(C0_SHARD_DIR))
    ap.add_argument("--skip-nonlinear", action="store_true")
    ap.add_argument("--force", action="store_true", help="refit behaviors with a checkpoint")
    ap.add_argument("--num-threads", type=int, default=8)
    args = ap.parse_args()
    # FIRST: --smoke re-execs with EPM_ISSUE763_SMOKE_SCOPE=1 (write paths
    # rebind under smoke_scope/); the env WITHOUT --smoke fails loud.
    ensure_smoke_scope(args.smoke)
    torch.set_num_threads(args.num_threads)

    if args.smoke:
        args.behaviors = args.behaviors[:1]
        args.n_perms = min(args.n_perms, 5)
        args.n_boot = min(args.n_boot, 25)
        args.k_random = min(args.k_random, 6)
        args.signflip_b = min(args.signflip_b, 500)
        args.nonlinear_b_perm = min(args.nonlinear_b_perm, 5)
        args.power_trials = min(args.power_trials, 2)
        smoke_dirs = COFIT_DIR / "smoke" / "pv_directions_v2"
        smoke_c0 = COFIT_DIR / "smoke" / "c0_shards"
        _write_smoke_surrogates(args.behaviors, smoke_dirs, smoke_c0)
        args.dirs_dir = str(smoke_dirs)
        args.c0_dir = str(smoke_c0)
    else:
        _stage_round_inputs(args.behaviors)

    # HARD exactness gates before any behavior is fit (batched-path discipline).
    # The cofit gate runs ON FIT_DEVICE (crash-fix r4): it builds its LayerCache
    # on the lane's device, so kernel_loco_preds — which derives its device from
    # the cache — is exercised on cuda at start-up; a device-threading miss on
    # the battery path fails HERE in seconds, not at battery time.
    gate_cofit = assert_cofit_matches_reference(device=FIT_DEVICE)
    gate_option_a = assert_option_a_contract()
    logger.info("[gate] cofit exactness %s | option-A contract %s", gate_cofit, gate_option_a)

    write_inputs_manifest(args.behaviors, smoke=args.smoke)

    _, instances = load_battery()
    fam_by_ctx = {inst["id"]: inst.get("family") for inst in instances}

    ckpt_dir = COFIT_DIR / "fit_by_behavior"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    records: dict[str, dict] = {}
    for behavior in args.behaviors:
        ckpt = ckpt_dir / f"{behavior}.json"
        if ckpt.exists() and not args.force:
            prior = load_json(ckpt)
            # input-aware resume: the key carries the direction/c0 sha256s, so a
            # checkpoint fit on superseded Phase-A/B artifacts never resume-skips.
            if prior.get("config") == _behavior_config_key(behavior, args):
                logger.info("[cofit] %s: matching checkpoint found — resume-skip", behavior)
                records[behavior] = prior
                continue
        rec = fit_behavior_cofit(behavior, args, fam_by_ctx)
        dump_json(rec, ckpt)  # checkpoint per behavior (intra-phase grain)
        records[behavior] = rec

    out = {
        "round": "neutral-contrast-and-cofit",
        "methods": list(METHODS),
        "method_labels": METHOD_LABELS,
        "by_behavior": records,
        "exactness_gates": {"cofit": gate_cofit, "option_a": gate_option_a},
        "fit_device": FIT_DEVICE,
        "metadata": reproducibility_metadata({"phase": "cofit_results"}),
    }
    dump_json(out, COFIT_DIR / "cofit_results.json")
    logger.info("[cofit] wrote %s", COFIT_DIR / "cofit_results.json")

    if not args.skip_nonlinear:
        nl = run_nonlinear_block(args.behaviors, records, args)
        dump_json(nl, COFIT_DIR / "nonlinear_tests.json")
        logger.info("[cofit] wrote %s", COFIT_DIR / "nonlinear_tests.json")

    print(f"[issue763.cofit] behaviors={args.behaviors} device={FIT_DEVICE} smoke={args.smoke}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
