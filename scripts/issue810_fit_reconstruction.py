#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ρ, ×, ², ȳ) in scientific docstrings + log messages.
"""Issue #810 DV (a) — reconstruction: held-out skill-over-mean R² per (layer × summary).

The linear context→answer map ``c_C[L] → summary[L]``, LOCO closed-form ridge
(primary) + batched 1-hidden MLP (validity), per (layer × summary), with a
selection-symmetric label-shuffle null. This is #722's DV asked over the #810
answer-side summary set: which summary's per-layer skill-over-mean R² beats the
``mean`` baseline (0.74-0.80 mid/late, #722) at mid/late layers?

SELF-CONTAINED against on-main primitives (plan §4.6). The per-position
``ridge_skill``-equivalent is RE-IMPLEMENTED inline over
``vectorized_mlp_skill.{robust_pca_basis, ridge_predict_loco_centered,
skill_over_mean_r2, loco_train_means, fit_batched_loco_mlp_multihead}`` — it does NOT
import ``scripts/issue722_per_position_vC_skill.py`` (stranded on branch
``fig-per-position``, ABSENT from main — a git-clone lane would ImportError,
built-but-stranded protection).

Free leg (Phase A, 0-GPU): mean/last/maxp summaries come from #658's stored
``store/v0_summaries.pt`` (already computed). Position summaries (im_end,
turn_nl, tail_k, head_k) come from #810's Phase B aligned-subset store.

Null (selection-symmetric, plan §6/§6.5): the ``summary`` target rows are
row-PERMUTED and re-fit; per (summary × layer) draws are persisted to
``null_matrix.json`` so the analyzer recomputes the honest max-over-{summary,
layer} band as a 0-GPU re-reduction.

Diagnostics folded in (Alternatives-lens binding Concerns, plan §6):
- ``summary[L] → summary[L]`` identity sanity-check (near-1 R² by construction;
  guards the re-implemented inline LOCO ridge against a silent bug).
- cosine(turn_nl activation, c_C activation) per layer across contexts (the
  boundary-triviality diagnostic — a format echo reads ≈1 + constant).

Usage::

    # free leg + fit (0-GPU CPU, after Phase B store lands on HF):
    uv run python scripts/issue810_fit_reconstruction.py \\
        --position-store-hf issue658_theory_assumptions/answer_position_sweep \\
        --out eval_results/issue_810

    # smoke (2 layers, ridge only, tiny synthetic-safe): --smoke
    uv run python scripts/issue810_fit_reconstruction.py --smoke \\
        --position-store-dir /tmp/i810_smoke/store --out /tmp/i810_smoke/out
"""

from __future__ import annotations

import argparse
import logging

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib
import sys
from collections import Counter
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402

# LOFO fold primitives from the committed adhoc LOFO benchmark script (main @
# 428f99e5a6, cherry-picked onto issue-810 — plan v11 §4.6 item 6).
from issue810_adhoc_lofo_heatmaps import (  # noqa: E402
    _recon_fold_predict,
    skill_over_mean_r2_lofo,
)
from issue810_batched_null import (  # noqa: E402
    batched_ridge_loco_null_skill,
    make_perm_matrix,
)
from issue810_common import (  # noqa: E402
    G1_ANSWER_POSITION_SWEEP_SUBDIR,
    G1_STORE_MANIFEST,
    G1_V0_SUMMARIES,
    GENRES,
    HF_DATA_REPO,
    HF_PREFIX,
    I594_CC_LAST_FILE,
    I594_PROBE_POOL_HASH,
    I658_STORE_MANIFEST,
    I658_V0_SUMMARIES,
    PCA_TARGET_DIM_CAP,
    SHUFFLE_NULL_PERMS,
    SHUFFLE_NULL_SEED,
    UH_SUMMARIES_HF_FILE,
    UH_SUMMARY_NAMES,
    assert_g1_probe_pool_hash,
    battery_family_map,
    context_ids_from_manifest,
    dump_json,
    enlarged_summary_names,
    load_json,
    reproducibility_metadata,
    summary_names,
    upload_out_dir,
)

# On-main fit primitives (the self-contained base — plan §4.6). NO import of the
# stranded fig-per-position script.
from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    MLPGroup,
    fit_batched_loco_mlp_multihead,
    ridge_predict_loco_centered,
    robust_pca_basis,
    skill_over_mean_r2,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue810_fit_reconstruction")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── inputs ────────────────────────────────────────────────────────────────────


_V0_BLOB_CACHE: dict[str, dict] = {}


def _load_v0_blob(genre: str) -> dict:
    """Load (and memoize) a genre's ``v0_summaries.pt`` pack; g1 is hash-pinned."""
    if genre not in _V0_BLOB_CACHE:
        from huggingface_hub import hf_hub_download

        v0_file = I658_V0_SUMMARIES if genre == "betley" else G1_V0_SUMMARIES
        p = hf_hub_download(HF_DATA_REPO, v0_file, repo_type="dataset")
        blob = torch.load(p, weights_only=False)
        if genre == "g1":
            assert_g1_probe_pool_hash(blob, G1_V0_SUMMARIES)
        _V0_BLOB_CACHE[genre] = blob
    return _V0_BLOB_CACHE[genre]


def _load_free_summaries(genre: str = "betley"):
    """{recipe: {ctx_id: (28,H) fp32}} for mean/last/maxp from the genre's v0_summaries.pt."""
    blob = _load_v0_blob(genre)
    return blob["summaries"], blob["capture_layers"]


def _load_cc(ctx_ids: list[str], capture_layers: list[int]) -> dict[str, np.ndarray]:
    """#594 last-input-token c_C, {ctx_id: (Lc,H) fp32}, probe_pool_hash pinned.

    BETLEY-ONLY (the #594 store is Betley-pinned by its own hash assert); the g1
    arm's c_C comes from ``_load_cc_for_genre('g1', ...)`` instead — the #658
    per-genre recomputed ``v0_summaries.pt::cc_last`` (plan v6 §4.6 item 3).
    """
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, I594_CC_LAST_FILE, repo_type="dataset")
    blob = torch.load(p, weights_only=False)
    pph = blob.get("probe_pool_hash")
    if pph != I594_PROBE_POOL_HASH:
        raise RuntimeError(f"#594 c_C probe_pool_hash drift: {pph} != {I594_PROBE_POOL_HASH}")
    tensor = blob["tensor"]  # (n_ctx, 28, H)
    iid_to_row = {iid: i for i, iid in enumerate(blob["instance_ids"])}
    missing = [c for c in ctx_ids if c not in iid_to_row]
    if missing:
        raise RuntimeError(f"c_C store missing {len(missing)} contexts: {missing[:5]}")
    return {c: tensor[iid_to_row[c]][capture_layers].float().numpy() for c in ctx_ids}


def _load_cc_for_genre(
    genre: str, ctx_ids: list[str], capture_layers: list[int]
) -> dict[str, np.ndarray]:
    """The reconstruction predictor c_C per genre, {ctx_id: (Lc,H) fp32}.

    betley → the #594 last-input-token HF store (``_load_cc``, hash-pinned to
    ``ad687bec…``). g1 → the g1 store's per-genre recomputed
    ``v0_summaries.pt::cc_last`` (hash-pinned to ``f277f8c3…`` via the pack's
    ``probe_pool_hash``; #658 ``--cc-last-from-store`` precedent) — the #594
    loader would smuggle Betley-pool context vectors into the g1 arm's input
    side. Fail loud on a missing key / missing contexts (never a silent skip).
    """
    if genre == "betley":
        return _load_cc(ctx_ids, capture_layers)
    blob = _load_v0_blob("g1")
    store_cc = blob.get("cc_last")
    if not store_cc:
        raise RuntimeError(
            "g1 v0_summaries.pt has no cc_last key — the store was built without "
            "--cc-recompute-last and cannot supply the per-genre c_C (plan v6 §4.6 item 3)"
        )
    missing = [c for c in ctx_ids if c not in store_cc]
    if missing:
        raise RuntimeError(f"g1 cc_last missing {len(missing)} contexts: {missing[:5]}")
    blob_layers = list(blob["capture_layers"])
    if blob_layers != list(capture_layers):
        raise RuntimeError(
            f"g1 cc_last capture_layers drift: store {blob_layers[:5]}... vs "
            f"requested {list(capture_layers)[:5]}..."
        )
    return {c: store_cc[c].float().numpy() for c in ctx_ids}


def _load_position_summaries(
    ctx_ids: list[str], hf_prefix: str | None, local_dir: Path | None
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, dict[str, int]]]:
    """{ctx_id: {position: (Lc,H) fp32}} + {ctx_id: coverage} from the Phase B store.

    Reads one ``<ctx>.pt`` at a time (per-context; the aligned-subset store is
    tiny at ~7 MB/ctx, but the per-context read keeps footprint bounded). Local
    dir is the smoke fast path; HF prefix is the production path.
    """
    from huggingface_hub import hf_hub_download

    out: dict[str, dict[str, np.ndarray]] = {}
    cov: dict[str, dict[str, int]] = {}
    for c in ctx_ids:
        if local_dir is not None and (local_dir / f"{c}.pt").is_file():
            blob = torch.load(local_dir / f"{c}.pt", weights_only=False)
        else:
            path = hf_hub_download(HF_DATA_REPO, f"{hf_prefix}/{c}.pt", repo_type="dataset")
            blob = torch.load(path, weights_only=False)
        names = blob["positions"]
        pv = blob["pos_vectors"].float().numpy()  # (n_pos, Lc, H)
        out[c] = {name: pv[i] for i, name in enumerate(names)}
        cov[c] = dict(blob["coverage"])
    return out, cov


# ── summary matrix assembly ───────────────────────────────────────────────────


def _summary_matrix(
    summary: str,
    layer_i: int,
    ctx_ids: list[str],
    free_summaries: dict,
    pos_summaries: dict,
    coverage: dict,
) -> tuple[np.ndarray, list[str]]:
    """(n_kept, H) summary matrix at one layer + the kept ctx_ids (coverage-aware).

    Free recipes (mean/last/maxp) read from #658's v0_summaries (always covered).
    Position recipes read from the Phase B store; a context whose coverage for
    this position is 0 (e.g. a deep tail_k on all-short answers) is DROPPED, and
    the kept-context list is returned so the predictor + null align to the same
    rows. Never a zero-filled silent row.
    """
    rows: list[np.ndarray] = []
    kept: list[str] = []
    for c in ctx_ids:
        if summary in ("mean", "last", "maxp"):
            rows.append(free_summaries[summary][c][layer_i].numpy())
            kept.append(c)
        else:
            if coverage[c].get(summary, 0) <= 0:
                continue
            rows.append(pos_summaries[c][summary][layer_i])
            kept.append(c)
    return (np.stack(rows) if rows else np.zeros((0, 1))), kept


# ── the inline per-position ridge_skill (re-implemented over on-main primitives) ──


def _fit_one_cell(Xc: np.ndarray, Yv: np.ndarray, pca_dim: int) -> tuple[dict, np.ndarray]:
    """Held-out ridge skill-over-mean R² for one (summary, layer) cell.

    Xc (n, H_c) = c_C at this layer; Yv (n, H) = the summary at this layer.
    Reduce Yv to its top-``pca_dim`` PCA basis (robust_pca_basis), fit LOCO ridge
    on the PCA target (ridge_predict_loco_centered), skill_over_mean_r2 on the
    PCA target. The MLP validity arm is NO LONGER fit here — it is batched ACROSS
    cells in ``main`` (one ``fit_batched_loco_mlp_multihead`` call per (n, d_in, p)
    shape group, the #722 vectorize-many-cell-fits mandate; the old per-cell
    one-group call defeated the batching). Returns ``(out, Y_pca)`` so the caller
    can batch the MLP arm over the same PCA target.

    This is the ``ridge_skill``-equivalent from the stranded fig-per-position
    script, re-implemented against the on-main primitives — target dim
    ``min(48, n-2)`` per plan §11.
    """
    n = Yv.shape[0]
    mu, comps, used_gesvd = robust_pca_basis(Yv, pca_dim)  # comps (k, H)
    Y_pca = (Yv - mu) @ comps.T  # (n, k) PCA target
    ridge_pred = ridge_predict_loco_centered(Xc, Y_pca)  # (n, k) held-out
    ridge = skill_over_mean_r2(ridge_pred, Y_pca)
    out = {
        "n": int(n),
        "pca_dim": int(comps.shape[0]),
        "used_gesvd_fallback": bool(used_gesvd),
        "ridge_skill": ridge["skill"],
        "ridge_median_per_dim_r2": ridge["median_per_dim_r2"],
    }
    return out, Y_pca


def _batch_mlp_validity(
    mlp_jobs: list[tuple[tuple, np.ndarray, np.ndarray]],
    device: str = "cpu",
    chunk_size: int = 256,
) -> dict:
    """Batch the MLP validity arm across ALL cells (grouped by (n, d_in, p) shape).

    ``mlp_jobs`` is a list of ``(cell_key, Xc, Y_pca)``. The fit uses
    ``fit_batched_loco_mlp_multihead`` — the production path per
    `.claude/rules/vectorize-many-cell-fits.md`: ONE shared-trunk net per
    (group, fold) predicting all ``p`` PCA dims jointly, E = n_groups × n_folds
    members (48× fewer at p=48 than the per-dim-scalar ``fit_batched_loco_mlp``,
    which needed 8-12h+ on an A100 at 1036 cells and was killed at the 4h
    watch-rule mark). The multihead ARCHITECTURE differs from the per-dim-scalar
    reference, so ``mlp_skill`` is the multihead variant's number — acceptable
    because this arm is a VALIDITY companion (does an MLP beat the ridge skill),
    not a bit-for-bit reproduction target. Groups in one call must share
    (n, d_in, p), so cells are bucketed by shape, one batched call per bucket
    (#722 mandate). Returns ``{cell_key: mlp_skill}``.

    ``chunk_size`` bounds members-per-optimizer-chunk and is REQUIRED here: the
    fitter's library default (4096) allocates a W1 chunk of
    (4096, hidden=512, d_in=3584) fp32 ≈ 30 GB — >100 GB with grads + the two
    AdamW moments — an OOM on both the A100-40 GPU lane and the 32 GB CPU
    instance. At 256 members the same footprint is ≈ 1.9 GB × 4 ≈ 7.5 GB
    (per-member W1 = 512×3584×4 B ≈ 7.34 MB; ×4 for param+grad+2 moments).
    """
    buckets: dict[tuple, list[MLPGroup]] = {}
    for key, Xc, Y_pca in mlp_jobs:
        shape = (Xc.shape[0], Xc.shape[1], Y_pca.shape[1])
        buckets.setdefault(shape, []).append(MLPGroup(key, Xc, Y_pca))
    y_by_key = {key: Y_pca for key, _Xc, Y_pca in mlp_jobs}
    mlp_skill_by_key: dict = {}
    for shape, groups in buckets.items():
        n, _d_in, _p = shape
        logger.info(
            "[phase=mlp] batching %d cells at shape %s (multihead: %d members, chunk_size=%d)",
            len(groups),
            shape,
            len(groups) * n,
            chunk_size,
        )
        res = fit_batched_loco_mlp_multihead(
            groups, seed=SHUFFLE_NULL_SEED, device=device, chunk_size=chunk_size
        )
        for g in groups:
            mlp = skill_over_mean_r2(res.preds_by_key[g.key], y_by_key[g.key])
            mlp_skill_by_key[g.key] = mlp["skill"]
    return mlp_skill_by_key


def _fit_null_draws(
    Xc: np.ndarray, Yv: np.ndarray, pca_dim: int, n_perms: int, seed: int, device: str = "cpu"
) -> list[float]:
    """Label-shuffle null: permute the summary rows, re-fit ridge, per-draw skill.

    Returns ``n_perms`` per-draw ridge skill-over-mean R² values (the
    selection-symmetric null distribution for this cell — the analyzer forms
    the honest max-over-{summary,layer} band by max-selecting these per draw).

    VECTORIZED (#722 mandate, `.claude/rules/vectorize-many-cell-fits.md`): the
    c_C design ``Xc`` is FIXED across permutations (only the PCA target rows are
    permuted), so every X-only LOCO-ridge factor (per-fold standardization, the
    dual Gram eigendecomposition, the per-λ dual-solve inverse) is computed ONCE
    and all ``n_perms`` permutations run as batched matmuls — NO Python-level
    per-perm re-fit loop. The batched closed form IS the serial refit (same PRESS
    / dual identities), so the per-draw skill is numerically identical to the old
    serial null (the smoke asserts this). The PCA basis is fit ONCE per cell (as
    before); the null permutes the PCA target rows exactly as the serial path did.
    """
    n = Yv.shape[0]
    mu, comps, _ = robust_pca_basis(Yv, pca_dim)
    Y_pca = (Yv - mu) @ comps.T
    rng = np.random.default_rng(seed)  # same per-cell seed as the serial reference
    perm = make_perm_matrix(n, n_perms, rng)  # (n_perms, n) — same draw order
    return batched_ridge_loco_null_skill(Xc, Y_pca, perm, device=device)


# ── diagnostics ───────────────────────────────────────────────────────────────


def _cosine_turn_nl_cc_per_layer(
    ctx_ids: list[str], capture_layers: list[int], cc: dict, pos_summaries: dict, coverage: dict
) -> dict:
    """Per-layer cosine(turn_nl activation, c_C activation) across contexts.

    The boundary-triviality diagnostic (Alternatives-lens binding Concern): if
    this cosine is ≈1 and roughly CONSTANT across contexts, a high turn_nl
    reconstruction R² is a shared boundary-constant echo, not context-specific
    answer content. Returns {layer: {mean_cos, std_cos, n}}.
    """
    out = {}
    for li, layer in enumerate(capture_layers):
        cosims = []
        for c in ctx_ids:
            if coverage[c].get("turn_nl", 0) <= 0:
                continue
            arr = pos_summaries[c]["turn_nl"]
            if li >= arr.shape[0] or arr[li].shape != cc[c][li].shape:
                # Mixed-model smoke only (0.5B position store vs 7B c_C): the
                # cosine is undefined across hidden sizes / layer counts — skip
                # EXPLICITLY. Production shapes always match (same 7B capture).
                continue
            a = arr[li]
            b = cc[c][li]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na < 1e-9 or nb < 1e-9:
                continue
            cosims.append(float(a @ b / (na * nb)))
        if cosims:
            out[layer] = {
                "mean_cos": float(np.mean(cosims)),
                "std_cos": float(np.std(cosims)),
                "n": len(cosims),
            }
    return out


def _identity_sanity_check(
    ctx_ids: list[str], free_summaries: dict, layer_i: int, pca_dim: int
) -> dict:
    """summary[L] → summary[L] LOCO fit should give near-1 skill-over-mean R².

    A cheap upper-anchor confirmation the re-implemented inline LOCO ridge
    reproduces the on-main primitives' behavior (guards a silent bug in the new
    loop). Uses the ``mean`` summary as X == Y.
    """
    rows = [free_summaries["mean"][c][layer_i].numpy() for c in ctx_ids]
    Y = np.stack(rows)
    res, _y_pca = _fit_one_cell(Y.copy(), Y.copy(), pca_dim)
    return {"layer_i": layer_i, "ridge_skill": res["ridge_skill"], "n": res["n"]}


# ── uh-round helpers: row tokens, LOFO point fits, null-join union bands ─────
# (plan v11 §4.6 item 3; the LOFO fold logic is REUSED from the committed
# issue810_adhoc_lofo_heatmaps.py — main @ 428f99e5a6, cherry-picked here.)


def _expand_rows(vals: list[str]) -> list[str]:
    """Expand --rows tokens ('uh-new', 'all-46') and validate against the axis."""
    tokens = {"uh-new": UH_SUMMARY_NAMES, "all-46": enlarged_summary_names()}
    out: list[str] = []
    for v in vals:
        out.extend(tokens.get(v, [v]))
    known = set(enlarged_summary_names())
    unknown = [r for r in out if r not in known]
    if unknown:
        raise SystemExit(f"unknown summary row(s) {unknown} — valid rows: the 46-row axis")
    return out


def _lofo_cell_skill(Xc: np.ndarray, Yv: np.ndarray, groups: list[str]) -> float | None:
    """7-family LOFO point skill for one (summary, layer) cell (no nulls).

    Mirrors the committed LOFO benchmark recipe exactly
    (``issue810_adhoc_lofo_heatmaps.build_panel3_reconstruction``): PCA target
    dim capped by the smallest train fold, per-fold train-only PCA basis +
    standardization + PRESS-λ ridge (``_recon_fold_predict``), variance-weighted
    held-out R² against the per-fold train-mean baseline.
    """
    if Yv.shape[0] < 8 or len(set(groups)) < 2:
        return None
    fam_counts = Counter(groups)
    min_train = len(groups) - max(fam_counts.values())
    pca_dim = min(PCA_TARGET_DIM_CAP, max(1, min_train - 2))
    preds, ys, bases = _recon_fold_predict(Xc, Yv, groups, pca_dim)
    r = skill_over_mean_r2_lofo(preds, ys, bases)
    return None if (isinstance(r, float) and np.isnan(r)) else float(r)


def _lofo_identity_ceiling(ctx_ids, free_summaries, layer_i: int, fam_map: dict) -> dict:
    """summary[L] → summary[L] LOFO fit: the fresh group-fold estimator ceiling."""
    rows = [free_summaries["mean"][c][layer_i].numpy() for c in ctx_ids]
    Y = np.stack(rows)
    groups = [fam_map[c] for c in ctx_ids]
    return {
        "layer_i": layer_i,
        "lofo_skill": _lofo_cell_skill(Y.copy(), Y.copy(), groups),
        "n": len(ctx_ids),
    }


def _committed_mean_skill_by_layer(committed_recon_path: str) -> dict[int, float]:
    """{layer: ridge_skill} for the committed round-1 `mean` benchmark row."""
    blob = load_json(committed_recon_path)
    out = {}
    for cell in blob["by_summary"]["mean"]:
        if cell.get("ridge_skill") is not None:
            out[int(cell["layer"])] = float(cell["ridge_skill"])
    if not out:
        raise RuntimeError(f"no mean-row skills in {committed_recon_path}")
    return out


def _parity_cells(args, cm: dict, cell_specs, y_of, ctx_ids_all, capture_layers) -> dict:
    """Recompute the given cells' per-draw nulls and byte-compare vs a committed matrix.

    ``cell_specs`` = [(summary, layer)]; ``y_of(summary, layer_index)`` returns
    the (n, H) target matrix over ``ctx_ids_all``. Shared engine of the
    per-matrix union-join parity gates (plan v11 §6 / v15 §4.6 item 3).
    """
    tolerance = float(args.parity_tolerance)
    gate_cc = _load_cc_for_genre(args.genre, list(ctx_ids_all), capture_layers)
    record: dict = {"tolerance": tolerance, "cells": [], "pass": True}
    for s, layer in cell_specs:
        li = capture_layers.index(layer)
        Yv = y_of(s, li)
        Xc = np.stack([gate_cc[c][li] for c in ctx_ids_all])
        cell_pca = min(PCA_TARGET_DIM_CAP, max(1, len(ctx_ids_all) - 2))
        draws = np.asarray(
            _fit_null_draws(Xc, Yv, cell_pca, args.n_perms, SHUFFLE_NULL_SEED, args.device),
            dtype=np.float64,
        )
        ref_list = cm.get(s, {}).get(str(layer))
        if ref_list is None or len(ref_list) != draws.shape[0]:
            raise RuntimeError(
                f"parity gate: committed matrix has no {args.n_perms}-draw cell for "
                f"{s}@L{layer} (got {None if ref_list is None else len(ref_list)})"
            )
        ref = np.asarray(ref_list, dtype=np.float64)
        max_abs = float(np.max(np.abs(draws - ref)))
        ok = bool(max_abs <= tolerance)
        record["cells"].append({"summary": s, "layer": layer, "max_abs_diff": max_abs, "pass": ok})
        record["pass"] = record["pass"] and ok
        logger.info(
            "[phase=parity_gate] %s@L%d max|Δ|=%.3e -> %s",
            s,
            layer,
            max_abs,
            "PASS" if ok else "FAIL",
        )
    return record


def _parity_gate(args, cm: dict, ctx_ids_all, capture_layers, free_summaries) -> dict:
    """Recompute 2 OLD cells' per-draw nulls and byte-compare vs the committed matrix.

    The union-join validity gate (plan v11 §6): valid ONLY if the permutation
    sequence is row-set-independent — ``_fit_null_draws`` seeds a fresh
    ``default_rng(seed)`` per cell (fact-checked, A5), so a like-seeded
    recompute of ``mean@L18`` / ``maxp@L21`` over the SAME 50 contexts must
    reproduce the committed vectors to ≤ ``tolerance`` abs. Gate cells are free
    rows (v0), so the full-manifest context set is used regardless of any
    position-store subset.
    """

    def y_of(s, li):
        return np.stack([free_summaries[s][c][li].numpy() for c in ctx_ids_all])

    return _parity_cells(args, cm, (("mean", 18), ("maxp", 21)), y_of, ctx_ids_all, capture_layers)


def _committed_best_layer(committed_recon_path: str, row: str) -> int:
    """The committed argmax-ridge_skill layer for one row (fail loud on absence)."""
    blob = load_json(committed_recon_path)
    cells = [c for c in blob["by_summary"].get(row, []) if c.get("ridge_skill") is not None]
    if not cells:
        raise RuntimeError(f"{committed_recon_path} has no fitted cells for row {row!r}")
    return int(max(cells, key=lambda c: float(c["ridge_skill"]))["layer"])


def _load_uh_pack_matrix(spec: str, ctx_ids_all, capture_layers) -> dict:
    """{row: {ctx: (Lc, H) fp32 np}} from the round-3 uh_summaries pack, validated.

    The Y source for the round-3 committed matrix's parity-gate cells + the
    parity-FAIL full-rerun fallback (plan v15 §4.6 item 3). Pack values are
    bit-identical to the round-3 position store the committed nulls were fit
    from (both are fp16 casts of the same in-forward fp32 probe-means —
    verified byte-equal at plan/implementation time). Production-validated via
    ``validate_uh_pack`` (a smoke-provenance pack refuses).
    """
    from issue810_common import validate_uh_pack
    from issue810_fit_readout import _load_uh_summaries  # deferred: readout imports THIS module

    uh_rows, uh_cov, meta = _load_uh_summaries(spec)
    validate_uh_pack(
        {r: {c: torch.from_numpy(v) for c, v in per.items()} for r, per in uh_rows.items()},
        uh_cov,
        meta,
        requested_rows=list(UH_SUMMARY_NAMES),
        ctx_ids=list(ctx_ids_all),
        expected_capture_layers=list(capture_layers),
    )
    return uh_rows


def _parity_gate_uh(args, cm: dict, ctx_ids_all, capture_layers, uh_rows: dict) -> dict:
    """2-cell parity gate for the round-3 committed matrix (plan v15 §4.6 item 3).

    Gate cells = ``uh_mean3`` + ``uh_nl`` at their COMMITTED best layers (read
    from ``--committed-recon-uh`` by argmax, never hardcoded); Y from the
    round-3 uh_summaries pack over the full manifest context set.
    """
    specs = tuple(
        (s, _committed_best_layer(args.committed_recon_uh, s)) for s in ("uh_mean3", "uh_nl")
    )

    def y_of(s, li):
        return np.stack([uh_rows[s][c][li] for c in ctx_ids_all])

    return _parity_cells(args, cm, specs, y_of, ctx_ids_all, capture_layers)


def _fallback_full_null_rerun(
    args,
    null_matrix_new: dict,
    ctx_ids,
    capture_layers,
    layers,
    free_summaries,
    pos_summaries,
    coverage,
    cc,
) -> dict:
    """Registered parity-FAIL fallback: recompute the OLD rows' nulls in-run.

    Never a silent mixed join (plan v11 §6/§8): every old row × fitted layer
    gets a fresh per-draw null AND a fresh point skill, the latter compared
    against the committed round-1 point skills (>1e-6 disagreement = the
    comparator itself is unstable → kill criterion 3, ``failure_class: data``).
    Old position rows read from the CURRENT (uh) store's recaptured old
    positions (drift-gated at capture time).
    """
    committed_points: dict | None = None
    if Path(args.committed_recon).is_file():
        committed_points = load_json(args.committed_recon)["by_summary"]
    union: dict[str, dict[str, list[float]]] = {}
    worst_point_diff = 0.0
    for s in summary_names():
        if s in null_matrix_new:
            continue
        union[s] = {}
        for li in layers:
            Yv, kept = _summary_matrix(s, li, ctx_ids, free_summaries, pos_summaries, coverage)
            if Yv.shape[0] < 4:
                continue
            Xc = np.stack([cc[c][li] for c in kept])
            cell_pca = min(PCA_TARGET_DIM_CAP, max(1, len(kept) - 2))
            union[s][str(capture_layers[li])] = _fit_null_draws(
                Xc, Yv, cell_pca, args.n_perms, SHUFFLE_NULL_SEED, args.device
            )
            fit, _ = _fit_one_cell(Xc, Yv, cell_pca)
            if committed_points is not None:
                ref_cells = {
                    int(c["layer"]): c for c in committed_points.get(s, []) if "layer" in c
                }
                ref_cell = ref_cells.get(capture_layers[li]) or {}
                ref = ref_cell.get("ridge_skill")
                ref_n = ref_cell.get("n_kept", ref_cell.get("n"))
                # Apples-to-apples ONLY over the SAME context set: a smoke's ctx
                # subset (e.g. n=12) legitimately differs from the committed
                # n=50 fit and is SKIPPED loudly, never compared.
                if ref is not None and ref_n == len(kept):
                    diff = abs(float(fit["ridge_skill"]) - float(ref))
                    worst_point_diff = max(worst_point_diff, diff)
                    if diff > 1e-6:
                        raise RuntimeError(
                            f"parity FAIL fallback: recomputed {s}@L{capture_layers[li]} "
                            f"point skill differs from the committed round-1 value by "
                            f"{diff:.2e} (>1e-6) — the round-1 comparator itself is "
                            "unstable (plan v11 kill criterion 3; failure_class: data)"
                        )
                elif ref is not None:
                    logger.warning(
                        "[phase=null_fallback] %s@L%d point-skill compare SKIPPED "
                        "(recompute n=%d != committed n=%s — ctx-subset run)",
                        s,
                        capture_layers[li],
                        len(kept),
                        ref_n,
                    )
        logger.info("[phase=null_fallback] %s re-nulled (%d layers)", s, len(union[s]))
    union.update({k: dict(v) for k, v in null_matrix_new.items()})
    logger.info(
        "[phase=null_fallback] full enlarged-grid rerun complete (worst point |Δ| %.2e)",
        worst_point_diff,
    )
    return union


def _null_join_and_bands(
    args,
    null_join_path: str,
    null_matrix_new: dict,
    results: dict,
    free_summaries: dict,
    pos_summaries: dict,
    coverage: dict,
    cc: dict,
    ctx_ids,
    ctx_ids_all,
    capture_layers,
    layers,
    diag: dict,
) -> dict:
    """Parity-gated union of committed + new per-draw nulls → enlarged-axis bands.

    Emits (plan v11 §6 registered reads, each `{statistic, band_97_5, ceiling,
    verdict}`):
    - ``enlarged_axis_max_selected`` — the union per-draw max band over ALL
      (summary × layer) cells vs the best observed NEW-row LOCO skill;
    - ``D_uh_difference`` (H1-uh) — D_uh = max over new rows × layers of
      (skill_new − committed mean benchmark), read against the per-draw
      new-minus-mean difference-matrix max (same max per draw).
    Ceilings: the LOCO identity ceiling recomputed in-run (diag) and, when
    present, the fresh LOFO identity ceiling. A band at/above the ceiling is
    verdicted failure-to-reject (never evidence of absence).
    """
    committed = load_json(null_join_path)
    if committed.get("seed") != SHUFFLE_NULL_SEED or committed.get("n_perms") != args.n_perms:
        raise RuntimeError(
            f"--null-join seed/n_perms mismatch: committed ({committed.get('seed')}, "
            f"{committed.get('n_perms')}) vs run ({SHUFFLE_NULL_SEED}, {args.n_perms})"
        )
    cm = committed["reconstruction"]
    parity = _parity_gate(args, cm, ctx_ids_all, capture_layers, free_summaries)
    if parity["pass"]:
        union: dict[str, dict[str, list[float]]] = {k: dict(v) for k, v in cm.items()}
        union.update({k: dict(v) for k, v in null_matrix_new.items()})
        mode = "union_join"
    else:
        logger.warning(
            "[phase=parity_gate] FAIL — flipping to the registered full enlarged-grid "
            "null rerun (plan v11 §6/§8 fallback; never a silent mixed join)"
        )
        union = _fallback_full_null_rerun(
            args,
            null_matrix_new,
            ctx_ids,
            capture_layers,
            layers,
            free_summaries,
            pos_summaries,
            coverage,
            cc,
        )
        mode = "full_rerun_fallback"

    return {
        "mode": mode,
        "parity_gate": parity,
        **_band_rows_from_union(args, union, null_matrix_new, results, diag),
        "union_matrix_new_rows_only": True,  # the committed rows stay in git round-1
    }


def _d_null_per_draw_max(args, union: dict, new_rows: list[str]) -> np.ndarray | None:
    """Per-draw D null: max over new (row, L) of (null_new[d] − null_mean[L][d]).

    The new-minus-mean difference matrices (plan v11 §6 read mode 2); the
    mean-row per-draw vectors come from the (joined or re-run) union itself.
    """
    mean_null = union.get("mean", {})
    diff_rows = []
    for s in new_rows:
        for lstr, draws in union.get(s, {}).items():
            if lstr in mean_null and len(draws) == args.n_perms:
                diff_rows.append(
                    np.asarray(draws, dtype=np.float64)
                    - np.asarray(mean_null[lstr], dtype=np.float64)
                )
    return np.stack(diff_rows).max(axis=0) if diff_rows else None


def _band_rows_from_union(
    args, union: dict, null_matrix_new: dict, results: dict, diag: dict
) -> dict:
    """The enlarged-axis band + D_uh reductions over a (joined or re-run) union."""
    # Per-draw max over the union axis (only complete n_perms-draw cells join).
    cells = []
    for s, per_layer in union.items():
        for lstr, draws in per_layer.items():
            if len(draws) == args.n_perms:
                cells.append(((s, lstr), np.asarray(draws, dtype=np.float64)))
    if not cells:
        raise RuntimeError("union band: no complete per-draw cells to reduce")
    stack = np.stack([d for _k, d in cells])  # (n_cells, n_perms)
    per_draw_max = stack.max(axis=0)  # (n_perms,)
    band_97_5 = float(np.percentile(per_draw_max, 97.5))

    # Ceilings (band-vs-ceiling standing lesson, plan v11 §6).
    loco_ceiling = max(
        (c["ridge_skill"] for c in diag["identity_sanity_check"] if c["ridge_skill"] is not None),
        default=None,
    )
    lofo_ceiling = None
    if diag.get("lofo_identity_ceiling"):
        lofo_ceiling = max(
            (c["lofo_skill"] for c in diag["lofo_identity_ceiling"] if c["lofo_skill"] is not None),
            default=None,
        )

    # Observed new-row skills + the D_uh difference statistic (H1-uh).
    new_rows = [s for s in null_matrix_new if s in set(UH_SUMMARY_NAMES)]
    obs_new: list[tuple[str, int, float]] = []
    for s in new_rows:
        for cell in results.get(s, []):
            if cell.get("ridge_skill") is not None:
                obs_new.append((s, int(cell["layer"]), float(cell["ridge_skill"])))
    mean_bench = _committed_mean_skill_by_layer(args.committed_recon)
    d_terms = [(s, L, v - mean_bench[L]) for (s, L, v) in obs_new if L in mean_bench]
    d_obs = max((t[2] for t in d_terms), default=None)
    d_arg = max(d_terms, key=lambda t: t[2]) if d_terms else None
    obs_max = max((v for (_s, _L, v) in obs_new), default=None)
    obs_arg = max(obs_new, key=lambda t: t[2]) if obs_new else None

    d_null_draws = _d_null_per_draw_max(args, union, new_rows)
    d_band = float(np.percentile(d_null_draws, 97.5)) if d_null_draws is not None else None

    def _verdict(stat, band, ceiling):
        if stat is None or band is None:
            return "not_computable"
        v = "clears_band" if stat > band else "within_band"
        if ceiling is not None and band >= ceiling:
            v += "; band_at_or_above_ceiling -> failure_to_reject"
        return v

    band_rows = {
        "enlarged_axis_max_selected": {
            "statistic": obs_max,
            "arg_cell": list(obs_arg) if obs_arg else None,
            "band_97_5": band_97_5,
            "ceiling": loco_ceiling,
            "verdict": _verdict(obs_max, band_97_5, loco_ceiling),
        },
        "D_uh_difference": {
            "statistic": d_obs,
            "arg_cell": list(d_arg) if d_arg else None,
            "band_97_5": d_band,
            # Max achievable difference under the estimator: ceiling − the best
            # committed mean benchmark (documented formula, not a fit).
            "ceiling": (
                (loco_ceiling - max(mean_bench.values())) if loco_ceiling is not None else None
            ),
            "effect_floor": 0.02,
            "meets_effect_floor": (d_obs is not None and d_obs >= 0.02),
            "verdict": _verdict(d_obs, d_band, None),
        },
        "lofo_identity_ceiling": lofo_ceiling,
    }
    return {
        "n_union_cells": len(cells),
        "union_rows": sorted({s for (s, _l), _d in cells}),
        "per_draw_max": [float(x) for x in per_draw_max],
        "d_null_draws": ([float(x) for x in d_null_draws] if d_null_draws is not None else None),
        "band_rows": band_rows,
    }


# ── multi-matrix union join (`_he` round, plan v15 §4.6 item 3) ──────────────

# The production 55-row union: 37 round-1 + 9 round-3 + 9 fresh empty rows.
UNION_EXPECTED_ROWS = 55
FRESH_MATRIX_ID = "fresh"


def _load_committed_matrix(args, path: str) -> tuple[str, dict]:
    """(matrix_id, per-draw matrix) for one committed --null-join path.

    ``matrix_id`` = the filename stem (stable, human-legible — e.g.
    ``null_matrix_reconstruction`` / ``null_matrix_user_header``); the
    seed/n_perms provenance is asserted per matrix.
    """
    committed = load_json(path)
    if committed.get("seed") != SHUFFLE_NULL_SEED or committed.get("n_perms") != args.n_perms:
        raise RuntimeError(
            f"--null-join {path}: seed/n_perms mismatch — committed "
            f"({committed.get('seed')}, {committed.get('n_perms')}) vs run "
            f"({SHUFFLE_NULL_SEED}, {args.n_perms})"
        )
    return Path(path).stem, committed["reconstruction"]


def _union_cells_from(cells: dict, matrix_id: str, matrix: dict, n_perms: int) -> None:
    """Fold one matrix's complete per-draw cells into the COMPOSITE-keyed union.

    Keys are ``(matrix_id, summary, layer_str)`` — collision-safe by
    construction (plan v15 Must-Fix): the 9 fresh empty rows reuse committed
    row NAMES (im_end/turn_nl collide with round-1; uh_*/bnd_* with round-3),
    so a bare-name-keyed join would silently overwrite committed rows and
    SHRINK the 55-row family (moving the max-selected band). A duplicate
    composite key fails loud.
    """
    for s, per_layer in matrix.items():
        for lstr, draws in per_layer.items():
            if len(draws) != n_perms:
                continue
            key = (matrix_id, s, str(lstr))
            if key in cells:
                raise RuntimeError(f"union join: duplicate composite key {key}")
            cells[key] = np.asarray(draws, dtype=np.float64)


def _fallback_full_rerun_multi(
    args, committed: dict, ctx_ids_all, capture_layers, free_summaries, uh_rows: dict
) -> dict:
    """Registered ANY-parity-FAIL fallback: recompute EVERY committed row's nulls in-run.

    Never a silent mixed join (plan v15 §6/§8): round-1 rows (free recipes from
    v0 + the 34 position rows from the ROUND-1 store, downloaded per-context)
    and round-3 rows (from the uh pack) each get a fresh per-draw null AND a
    fresh point skill compared against their committed recon JSONs (>1e-6
    disagreement = the comparator itself is unstable → kill criterion,
    ``failure_class: data``). Returns the composite-keyed committed-cells dict
    (fresh rows are folded in by the caller).
    """
    r1_pos, r1_cov = _load_position_summaries(
        list(ctx_ids_all), f"{HF_PREFIX}/answer_position_sweep", None
    )
    committed_points = {
        "round1": load_json(args.committed_recon)["by_summary"],
        "round3": load_json(args.committed_recon_uh)["by_summary"],
    }
    cells: dict = {}
    worst_point_diff = 0.0
    for mid, cm in committed.items():
        source = "round1" if "mean" in cm else "round3"
        points = committed_points[source]
        for s, per_layer in cm.items():
            for lstr in per_layer:
                layer = int(lstr)
                li = capture_layers.index(layer)
                if source == "round3":
                    Yv = np.stack([uh_rows[s][c][li] for c in ctx_ids_all])
                    kept = list(ctx_ids_all)
                else:
                    Yv, kept = _summary_matrix(
                        s, li, list(ctx_ids_all), free_summaries, r1_pos, r1_cov
                    )
                if Yv.shape[0] < 4:
                    continue
                gate_cc = _load_cc_for_genre(args.genre, kept, capture_layers)
                Xc = np.stack([gate_cc[c][li] for c in kept])
                cell_pca = min(PCA_TARGET_DIM_CAP, max(1, len(kept) - 2))
                cells[(mid, s, lstr)] = np.asarray(
                    _fit_null_draws(Xc, Yv, cell_pca, args.n_perms, SHUFFLE_NULL_SEED, args.device),
                    dtype=np.float64,
                )
                fit, _ = _fit_one_cell(Xc, Yv, cell_pca)
                ref_cells = {int(c["layer"]): c for c in points.get(s, []) if "layer" in c}
                ref_cell = ref_cells.get(layer) or {}
                ref = ref_cell.get("ridge_skill")
                ref_n = ref_cell.get("n_kept", ref_cell.get("n"))
                if ref is not None and ref_n == len(kept):
                    diff = abs(float(fit["ridge_skill"]) - float(ref))
                    worst_point_diff = max(worst_point_diff, diff)
                    if diff > 1e-6:
                        raise RuntimeError(
                            f"multi-join parity FAIL fallback: recomputed {s}@L{layer} "
                            f"({source}) point skill differs from committed by {diff:.2e} "
                            "(>1e-6) — the comparator itself is unstable (plan v15 kill "
                            "criterion; failure_class: data)"
                        )
                elif ref is not None:
                    logger.warning(
                        "[phase=null_fallback] %s@L%d point compare SKIPPED "
                        "(n=%d != committed n=%s)",
                        s,
                        layer,
                        len(kept),
                        ref_n,
                    )
            logger.info("[phase=null_fallback] %s/%s re-nulled", mid, s)
    logger.info(
        "[phase=null_fallback] full %d-cell committed rerun complete (worst point |Δ| %.2e)",
        len(cells),
        worst_point_diff,
    )
    return cells


def _band_rows_from_union_multi(
    args, cells: dict, committed: dict, null_matrix_new: dict, results: dict, diag: dict
) -> dict:
    """Enlarged-axis band + difference read over the COMPOSITE-keyed union.

    Mirrors ``_band_rows_from_union`` on ``(matrix_id, summary, layer)`` keys:
    the max-selected band is over ALL union cells; the observed statistic is
    the best FRESH empty-row LOCO skill; the difference read (``D_he``, the
    v11-shape companion — the v15 PRIMARY paired read lives in
    ``paired_full_minus_empty.json``) subtracts the round-1 ``mean`` row's
    per-draw vectors per layer. Every band row carries the recomputed ceiling.
    """
    if not cells:
        raise RuntimeError("multi union band: no complete per-draw cells to reduce")
    stack = np.stack(list(cells.values()))  # (n_cells, n_perms)
    per_draw_max = stack.max(axis=0)
    band_97_5 = float(np.percentile(per_draw_max, 97.5))

    loco_ceiling = max(
        (c["ridge_skill"] for c in diag["identity_sanity_check"] if c["ridge_skill"] is not None),
        default=None,
    )
    lofo_ceiling = None
    if diag.get("lofo_identity_ceiling"):
        lofo_ceiling = max(
            (c["lofo_skill"] for c in diag["lofo_identity_ceiling"] if c["lofo_skill"] is not None),
            default=None,
        )

    new_rows = sorted(null_matrix_new)
    obs_new: list[tuple[str, int, float]] = []
    for s in new_rows:
        for cell in results.get(s, []):
            if cell.get("ridge_skill") is not None:
                obs_new.append((s, int(cell["layer"]), float(cell["ridge_skill"])))
    obs_max = max((v for (_s, _L, v) in obs_new), default=None)
    obs_arg = max(obs_new, key=lambda t: t[2]) if obs_new else None

    mean_bench = _committed_mean_skill_by_layer(args.committed_recon)
    d_terms = [(s, L, v - mean_bench[L]) for (s, L, v) in obs_new if L in mean_bench]
    d_obs = max((t[2] for t in d_terms), default=None)
    d_arg = max(d_terms, key=lambda t: t[2]) if d_terms else None
    # Per-draw D null: fresh-row draws minus the round-1 mean row's draws.
    mean_mid = next((mid for mid, cm in committed.items() if "mean" in cm), None)
    diff_rows = []
    for s in new_rows:
        for lstr, draws in null_matrix_new.get(s, {}).items():
            mean_key = (mean_mid, "mean", str(lstr))
            if mean_mid is not None and mean_key in cells and len(draws) == args.n_perms:
                diff_rows.append(np.asarray(draws, dtype=np.float64) - cells[mean_key])
    d_null_draws = np.stack(diff_rows).max(axis=0) if diff_rows else None
    d_band = float(np.percentile(d_null_draws, 97.5)) if d_null_draws is not None else None

    def _verdict(stat, band, ceiling):
        if stat is None or band is None:
            return "not_computable"
        v = "clears_band" if stat > band else "within_band"
        if ceiling is not None and band >= ceiling:
            v += "; band_at_or_above_ceiling -> failure_to_reject"
        return v

    band_rows = {
        "enlarged_axis_max_selected": {
            "statistic": obs_max,
            "arg_cell": list(obs_arg) if obs_arg else None,
            "band_97_5": band_97_5,
            "ceiling": loco_ceiling,
            "verdict": _verdict(obs_max, band_97_5, loco_ceiling),
        },
        "D_he_difference": {
            "statistic": d_obs,
            "arg_cell": list(d_arg) if d_arg else None,
            "band_97_5": d_band,
            "ceiling": (
                (loco_ceiling - max(mean_bench.values())) if loco_ceiling is not None else None
            ),
            "note": (
                "v11-shape companion (best empty row minus the committed mean benchmark); "
                "the v15 PRIMARY read is the paired bootstrap in paired_full_minus_empty.json"
            ),
            "verdict": _verdict(d_obs, d_band, None),
        },
        "lofo_identity_ceiling": lofo_ceiling,
    }
    return {
        "n_union_cells": len(cells),
        "union_rows": sorted({f"{mid}::{s}" for (mid, s, _l) in cells}),
        "union_key_schema": "(matrix_id, summary, layer)",
        "per_draw_max": [float(x) for x in per_draw_max],
        "d_null_draws": ([float(x) for x in d_null_draws] if d_null_draws is not None else None),
        "band_rows": band_rows,
    }


def _null_join_and_bands_multi(
    args,
    null_matrix_new: dict,
    results: dict,
    free_summaries: dict,
    ctx_ids_all,
    capture_layers,
    diag: dict,
) -> dict:
    """Multi-matrix parity-gated union (plan v15 §4.6 item 3 / §6).

    Each committed matrix gets its OWN 2-cell byte-parity gate (round-1:
    mean@L18 + maxp@L21 from v0; round-3: uh_mean3 + uh_nl at their committed
    best layers from the uh pack); ANY gate FAIL flips to the registered full
    committed-rows rerun (never a mixed join). The union is COMPOSITE-keyed
    ``(matrix_id, summary, layer)`` and asserts unique keys AND — on a
    production run — exactly 55 rows × 28 layers == 1540 cells BEFORE any band
    is emitted (a name-keyed join would silently shrink the family).
    """
    committed: dict[str, dict] = {}
    for path in args.null_join:
        mid, cm = _load_committed_matrix(args, path)
        if mid in committed:
            raise RuntimeError(f"--null-join: duplicate matrix id {mid!r} (paths must differ)")
        committed[mid] = cm
    uh_rows = _load_uh_pack_matrix(args.uh_summaries, ctx_ids_all, capture_layers)
    parities: dict[str, dict] = {}
    for mid, cm in committed.items():
        if "mean" in cm:
            parities[mid] = _parity_gate(args, cm, ctx_ids_all, capture_layers, free_summaries)
        elif "uh_mean3" in cm:
            parities[mid] = _parity_gate_uh(args, cm, ctx_ids_all, capture_layers, uh_rows)
        else:
            raise RuntimeError(
                f"--null-join {mid}: unrecognized committed matrix (no registered gate cells — "
                f"rows {sorted(cm)[:5]}...)"
            )
    all_pass = all(p["pass"] for p in parities.values())
    cells: dict = {}
    if all_pass:
        for mid, cm in committed.items():
            _union_cells_from(cells, mid, cm, args.n_perms)
        mode = "union_join_multi"
    else:
        logger.warning(
            "[phase=parity_gate] FAIL on >=1 matrix — flipping to the registered full "
            "committed-rows null rerun (plan v15 §6/§8 fallback; never a mixed join)"
        )
        cells = _fallback_full_rerun_multi(
            args, committed, ctx_ids_all, capture_layers, free_summaries, uh_rows
        )
        mode = "full_rerun_fallback_multi"
    _union_cells_from(cells, FRESH_MATRIX_ID, null_matrix_new, args.n_perms)
    n_rows = len({(mid, s) for (mid, s, _l) in cells})
    n_bare_names = len({s for (_m, s, _l) in cells})
    if not args.smoke:
        expected_cells = UNION_EXPECTED_ROWS * len(capture_layers)
        if n_rows != UNION_EXPECTED_ROWS:
            raise RuntimeError(
                f"union row count {n_rows} != {UNION_EXPECTED_ROWS} (37 round-1 + 9 round-3 "
                f"+ 9 fresh; bare names {n_bare_names}) — a shrunken family moves the "
                "max-selected band (plan v15 Must-Fix)"
            )
        if len(cells) != expected_cells:
            raise RuntimeError(
                f"n_union_cells {len(cells)} != {UNION_EXPECTED_ROWS} * "
                f"{len(capture_layers)} == {expected_cells} (plan v15 Must-Fix)"
            )
    else:
        logger.info(
            "[phase=union] smoke: %d composite rows / %d cells (production asserts 55/1540)",
            n_rows,
            len(cells),
        )
    return {
        "mode": mode,
        "parity_gates": parities,
        "parity_gate": {"pass": all_pass},  # sentinel-compat summary field
        **_band_rows_from_union_multi(args, cells, committed, null_matrix_new, results, diag),
        "union_matrix_new_rows_only": True,  # committed rows stay in git rounds 1/3
    }


# ── main ──────────────────────────────────────────────────────────────────────


def _fit_grid(
    args,
    summaries: list[str],
    layers: list[int],
    ctx_ids: list[str],
    capture_layers: list[int],
    free_summaries: dict,
    pos_summaries: dict,
    coverage: dict,
    cc: dict,
    fam_map: dict | None,
    do_loco: bool,
    do_lofo: bool,
):
    """The (summary × layer) fit grid: LOCO ridge (+nulls, +MLP jobs) and/or LOFO points.

    Extracted verbatim from ``main`` (behavior identical for the default
    ``--fold-family loco``): per cell, coverage-aware matrix assembly, LOCO
    ``_fit_one_cell`` + per-cell 1000-perm batched null, LOFO 7-family point
    skill when requested (no nulls — the registered ordering-only companion).
    Returns ``(results, null_matrix, mlp_jobs, cell_ref)``.
    """
    results: dict[str, list[dict]] = {}
    # Selection-symmetric null: per (summary × layer) per-draw skill matrix.
    null_matrix: dict[str, dict[str, list[float]]] = {}
    # MLP validity jobs collected across ALL cells, batched by shape AFTER the
    # ridge loop (the #722 vectorize-many-cell-fits mandate) — cell_key ->
    # (summary, layer_i) so mlp_skill can be attached back to the right cell dict.
    mlp_jobs: list[tuple[tuple, np.ndarray, np.ndarray]] = []
    cell_ref: dict[tuple, dict] = {}
    for summary in summaries:
        cells: list[dict] = []
        null_matrix[summary] = {}
        for li in layers:
            Yv, kept = _summary_matrix(
                summary, li, ctx_ids, free_summaries, pos_summaries, coverage
            )
            if Yv.shape[0] < 4:
                cells.append(
                    {"layer": capture_layers[li], "n": int(Yv.shape[0]), "ridge_skill": None}
                )
                continue
            Xc = np.stack([cc[c][li] for c in kept])
            cell_pca = min(PCA_TARGET_DIM_CAP, max(1, len(kept) - 2))
            if do_loco:
                fit, y_pca = _fit_one_cell(Xc, Yv, cell_pca)
                fit["layer"] = capture_layers[li]
                fit["n_kept"] = len(kept)
            else:
                # LOFO-only point-fit mode: no LOCO ridge, no nulls, no MLP arm
                # (plan v11 §4.6 item 3 — LOFO is the ordering-only companion).
                fit = {"layer": capture_layers[li], "n_kept": len(kept), "n": len(kept)}
            if do_lofo:
                fit["lofo_skill"] = _lofo_cell_skill(Xc, Yv, [fam_map[c] for c in kept])
            cells.append(fit)
            if do_loco and not args.no_mlp:
                key = (summary, capture_layers[li])
                mlp_jobs.append((key, Xc, y_pca))
                cell_ref[key] = fit
            if do_loco and args.n_perms > 0:
                # --n-perms 0 (the `_btdr` no-nulls path, plan v18 §4.6 item 3):
                # skip the permutation battery + band rows entirely — LOCO+LOFO
                # point skills only. Unguarded, make_perm_matrix(n, 0, rng)
                # crashes via np.stack([]) (issue810_batched_null.py:75).
                null_matrix[summary][str(capture_layers[li])] = _fit_null_draws(
                    Xc, Yv, cell_pca, args.n_perms, SHUFFLE_NULL_SEED, device=args.device
                )
        results[summary] = cells
        logger.info("[phase=fit] %s done (%d layers)", summary, len(cells))
    return results, null_matrix, mlp_jobs, cell_ref


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #810 DV (a): reconstruction skill-over-mean R²")
    ap.add_argument(
        "--genre",
        choices=list(GENRES),
        default="betley",
        help="probe-corpus genre: 'betley' (default — the parent's sources, bit-for-bit: "
        "#658 v0_summaries + the #594 c_C store) or 'g1' (#658's UltraChat arm: g1 "
        "v0_summaries + the g1 store's per-genre cc_last; both f277f8c3-hash-pinned)",
    )
    ap.add_argument(
        "--position-store-hf",
        default=None,
        help="HF prefix of the Phase B aligned-subset store (default: the genre's — "
        "answer_position_sweep for betley, answer_position_sweep_<genre-tag> for g1)",
    )
    ap.add_argument("--position-store-dir", default=None, help="local Phase B store (smoke)")
    ap.add_argument(
        "--free-only",
        action="store_true",
        help="Phase A free leg: fit only mean/last/maxp over ALL 50 manifest "
        "contexts (no Phase B position store needed) — runs before Phase B lands",
    )
    ap.add_argument("--out", default=str(PROJECT_ROOT / "eval_results" / "issue_810"))
    ap.add_argument(
        "--summaries",
        "--rows",
        nargs="*",
        default=None,
        help="subset of summary rows to fit (default = all; parent behavior). Accepts the "
        "tokens 'uh-new' (the 9 new user-header rows) and 'all-46' (the enlarged axis). "
        "`--rows` is the plan-v11 alias.",
    )
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="subset of layer indices")
    ap.add_argument(
        "--fold-family",
        choices=["loco", "lofo", "both"],
        default="loco",
        help="outer CV: 'loco' (parent behavior, byte-for-bit; nulls + MLP arm), 'lofo' "
        "(7-family point fits ONLY — no nulls, the plan-v11 registered ordering-only "
        "companion), or 'both'",
    )
    ap.add_argument(
        "--null-join",
        nargs="*",
        default=None,
        help="path(s) to COMMITTED per-draw recon null matrices. ONE path (round-3 "
        "behavior, byte-for-bit): recomputes 2 OLD cells (mean@L18, maxp@L21) with the "
        "same seed and byte-compares (<=1e-12) their per-draw vectors as the "
        "union-validity parity gate; PASS joins committed + new rows into the "
        "enlarged-axis union band; FAIL falls back to a full enlarged-grid null rerun "
        "(registered fallback, plan v11 §6/§8) — never a silent mixed join. MULTIPLE "
        "paths (the `_he` round, plan v15 §4.6 item 3): each matrix gets its OWN 2-cell "
        "gate (round-1: mean@L18/maxp@L21; round-3: uh_mean3/uh_nl at committed best "
        "layers, Y from --uh-summaries) and the union is COMPOSITE-keyed "
        "(matrix_id, summary, layer) with the 55-row/1540-cell production assert.",
    )
    ap.add_argument(
        "--committed-recon-uh",
        default=str(
            PROJECT_ROOT
            / "eval_results"
            / "issue_810"
            / "user-header-newline-summary"
            / "reconstruction_skill_user_header.json"
        ),
        help="the committed round-3 recon point-skill JSON (gate-cell best layers + the "
        "multi-join fallback point-skill comparator for the round-3 rows)",
    )
    ap.add_argument(
        "--uh-summaries",
        default=UH_SUMMARIES_HF_FILE,
        help="round-3 uh_summaries pack (local path or HF data-repo path) — the Y source "
        "for the round-3 matrix's parity-gate cells + the multi-join fallback rerun",
    )
    ap.add_argument(
        "--parity-tolerance",
        type=float,
        default=1e-12,
        help="max abs per-draw difference for the --null-join parity gate (default 1e-12 — "
        "the plan-v11 byte-parity bar for a same-device-class recompute; a cross-device "
        "run, e.g. the CPU smoke against the CUDA-committed matrix, legitimately sits at "
        "~4e-7 and exercises the FAIL->full-rerun fallback unless this is raised)",
    )
    ap.add_argument(
        "--committed-recon",
        default=str(
            PROJECT_ROOT / "eval_results" / "issue_810" / "reconstruction_skill_by_summary.json"
        ),
        help="the committed round-1 recon point-skill JSON (the observed `mean` benchmark "
        "for the D_uh difference statistic + the fallback point-skill comparator)",
    )
    ap.add_argument(
        "--out-suffix",
        default=None,
        help="output filename tag: reconstruction_skill_<suffix>.json + "
        "null_matrix_<suffix>.json (default None = the parent filenames, byte-for-bit)",
    )
    ap.add_argument("--n-perms", type=int, default=SHUFFLE_NULL_PERMS)
    ap.add_argument("--no-mlp", action="store_true", help="skip the MLP validity arm")
    ap.add_argument(
        "--mlp-chunk-size",
        type=int,
        default=256,
        help="members per optimizer chunk for the multihead MLP validity arm; bounds peak "
        "memory (256 -> ~7.5 GB at d_in=3584/hidden=512 incl. grads+AdamW moments; the "
        "library default 4096 would be ~120 GB and OOM the A100-40 / 32 GB CPU lanes)",
    )
    ap.add_argument(
        "--device",
        default="cpu",
        help="torch device for the batched null + MLP validity fits ('cpu' default, "
        "'cuda' to run on a GPU lane — CPU behavior is byte-identical to the default)",
    )
    ap.add_argument(
        "--upload-prefix",
        default=None,
        help="HF data-repo path prefix (e.g. 'issue810/phase_d_recon') to bulk-upload the "
        "out-dir *.json to after the fit; UNSET (default) = no upload (today's behavior). "
        "Set on an ephemeral GCP lane so the results survive instance teardown.",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # Fail fast: --device cuda with no CUDA never silently falls back to CPU.
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    from huggingface_hub import hf_hub_download

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.position_store_hf is None:
        args.position_store_hf = (
            f"{HF_PREFIX}/answer_position_sweep"
            if args.genre == "betley"
            else f"{HF_PREFIX}/{G1_ANSWER_POSITION_SWEEP_SUBDIR}"
        )

    logger.info(
        "[phase=load] manifest + free summaries + c_C + position store (genre=%s)", args.genre
    )
    manifest_file = I658_STORE_MANIFEST if args.genre == "betley" else G1_STORE_MANIFEST
    man = load_json(hf_hub_download(HF_DATA_REPO, manifest_file, repo_type="dataset"))
    if args.genre == "g1":
        assert_g1_probe_pool_hash(man, G1_STORE_MANIFEST)
    ctx_ids_all = context_ids_from_manifest(man)
    free_summaries, capture_layers = _load_free_summaries(args.genre)

    # --free-only (Phase A): mean/last/maxp over ALL 50 manifest contexts; no
    # Phase B position store. Otherwise restrict to the position store's contexts.
    local_dir = Path(args.position_store_dir) if args.position_store_dir else None
    if args.free_only:
        ctx_ids = list(ctx_ids_all)
        pos_summaries, coverage = {c: {} for c in ctx_ids}, {c: {} for c in ctx_ids}
    else:
        if local_dir is not None:
            pos_man = load_json(local_dir / "manifest.json")
        else:
            pos_man = load_json(
                hf_hub_download(
                    HF_DATA_REPO, f"{args.position_store_hf}/manifest.json", repo_type="dataset"
                )
            )
        ctx_ids = [c for c in pos_man["context_ids"] if c in ctx_ids_all]
        pos_summaries, coverage = _load_position_summaries(
            ctx_ids, args.position_store_hf, local_dir
        )
    logger.info("contexts: %d (free_only=%s)", len(ctx_ids), args.free_only)

    cc = _load_cc_for_genre(args.genre, ctx_ids, capture_layers)

    default_summaries = ["mean", "last", "maxp"] if args.free_only else summary_names()
    summaries = _expand_rows(args.summaries) if args.summaries else default_summaries
    layers = args.layers if args.layers is not None else list(range(len(capture_layers)))
    n = len(ctx_ids)
    pca_dim = min(PCA_TARGET_DIM_CAP, max(1, n - 2))
    do_loco = args.fold_family in ("loco", "both")
    do_lofo = args.fold_family in ("lofo", "both")
    fam_map = battery_family_map() if do_lofo else None
    if fam_map is not None:
        missing_fam = [c for c in ctx_ids if c not in fam_map]
        if missing_fam:
            raise RuntimeError(f"{len(missing_fam)} contexts have no family: {missing_fam[:5]}")
    logger.info(
        "fitting %d summaries x %d layers (n=%d, pca_dim=%d, fold_family=%s)",
        len(summaries),
        len(layers),
        n,
        pca_dim,
        args.fold_family,
    )

    results, null_matrix, mlp_jobs, cell_ref = _fit_grid(
        args,
        summaries,
        layers,
        ctx_ids,
        capture_layers,
        free_summaries,
        pos_summaries,
        coverage,
        cc,
        fam_map,
        do_loco,
        do_lofo,
    )

    # Batch the MLP validity arm across all cells (one call per shape group), then
    # attach mlp_skill back to each cell. Batching (not one-group-per-call) is the
    # #722 fix — the on-main fit_batched_loco_mlp_multihead batches (group × fold)
    # with a joint p-output head.
    if mlp_jobs:
        mlp_skill_by_key = _batch_mlp_validity(
            mlp_jobs, device=args.device, chunk_size=args.mlp_chunk_size
        )
        for key, skill in mlp_skill_by_key.items():
            cell_ref[key]["mlp_skill"] = skill

    # Diagnostics
    diag = {
        "cosine_turn_nl_cc_per_layer": _cosine_turn_nl_cc_per_layer(
            ctx_ids, capture_layers, cc, pos_summaries, coverage
        ),
        "identity_sanity_check": [
            _identity_sanity_check(ctx_ids, free_summaries, li, pca_dim) for li in layers
        ],
    }
    if do_lofo:
        # Fresh LOFO identity ceiling (plan v11 §4.6 item 3 — a NEW number,
        # computed once per run; the LOCO 0.857 does not bound the group-fold).
        diag["lofo_identity_ceiling"] = [
            _lofo_identity_ceiling(ctx_ids, free_summaries, li, fam_map) for li in layers
        ]

    # Enlarged-axis union band via --null-join (plan v11 §4.6 item 3 / §6):
    # per-matrix parity gate on 2 OLD cells, then join committed + new per-draw
    # matrices; FAIL → registered full rerun fallback (never a mixed join).
    # ONE path = the round-3 single-matrix behavior byte-for-bit; MULTIPLE
    # paths = the `_he` composite-keyed 55-row union (plan v15 §4.6 item 3).
    union_band: dict | None = None
    if args.null_join:
        if len(args.null_join) == 1:
            union_band = _null_join_and_bands(
                args,
                args.null_join[0],
                null_matrix,
                results,
                free_summaries,
                pos_summaries,
                coverage,
                cc,
                ctx_ids,
                ctx_ids_all,
                capture_layers,
                layers,
                diag,
            )
        else:
            union_band = _null_join_and_bands_multi(
                args,
                null_matrix,
                results,
                free_summaries,
                ctx_ids_all,
                capture_layers,
                diag,
            )

    recon_name = (
        f"reconstruction_skill_{args.out_suffix}.json"
        if args.out_suffix
        else "reconstruction_skill_by_summary.json"
    )
    null_name = (
        f"null_matrix_{args.out_suffix}.json"
        if args.out_suffix
        else "null_matrix_reconstruction.json"
    )
    dump_json(
        {
            "dv": "reconstruction_skill_over_mean_r2",
            "genre": args.genre,
            "predictor": (
                "cc_last_input_token (#594)"
                if args.genre == "betley"
                else "cc_last_input_token (g1 store v0_summaries.pt::cc_last, per-genre)"
            ),
            "n_contexts": n,
            "pca_target_dim": pca_dim,
            "capture_layers": capture_layers,
            "fold_family": args.fold_family,
            "by_summary": results,
            "diagnostics": diag,
            "band_rows": (union_band or {}).get("band_rows"),
            "reproducibility": reproducibility_metadata(),
            "smoke": args.smoke,
        },
        out_dir / recon_name,
    )
    # Null matrix persisted separately (its own primary-deliverable file; §6.5).
    dump_json(
        {
            "dv": "reconstruction",
            "genre": args.genre,
            "axes": "summary -> layer -> [per-draw ridge skill]",
            "n_perms": args.n_perms,
            "seed": SHUFFLE_NULL_SEED,
            "reconstruction": null_matrix,
            "union_band": union_band,
        },
        out_dir / null_name,
    )
    if args.upload_prefix:
        logger.info("[phase=upload] fit-result JSONs -> %s", args.upload_prefix)
        landed = upload_out_dir(out_dir, args.upload_prefix)
        logger.info("[phase=upload] verified fit-result JSONs under %s/", landed)
    logger.info("[phase=done] wrote reconstruction results + null matrix to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
