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
skill_over_mean_r2, loco_train_means, fit_batched_loco_mlp}`` — it does NOT
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
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from issue810_batched_null import (  # noqa: E402
    batched_ridge_loco_null_skill,
    make_perm_matrix,
)
from issue810_common import (  # noqa: E402
    HF_DATA_REPO,
    I594_CC_LAST_FILE,
    I594_PROBE_POOL_HASH,
    I658_STORE_MANIFEST,
    I658_V0_SUMMARIES,
    PCA_TARGET_DIM_CAP,
    SHUFFLE_NULL_PERMS,
    SHUFFLE_NULL_SEED,
    context_ids_from_manifest,
    dump_json,
    load_json,
    reproducibility_metadata,
    summary_names,
)

# On-main fit primitives (the self-contained base — plan §4.6). NO import of the
# stranded fig-per-position script.
from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    MLPGroup,
    fit_batched_loco_mlp,
    ridge_predict_loco_centered,
    robust_pca_basis,
    skill_over_mean_r2,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue810_fit_reconstruction")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# ── inputs ────────────────────────────────────────────────────────────────────


def _load_free_summaries():
    """{recipe: {ctx_id: (28,H) fp32}} for mean/last/maxp from #658 v0_summaries.pt."""
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(HF_DATA_REPO, I658_V0_SUMMARIES, repo_type="dataset")
    blob = torch.load(p, weights_only=False)
    return blob["summaries"], blob["capture_layers"]


def _load_cc(ctx_ids: list[str], capture_layers: list[int]) -> dict[str, np.ndarray]:
    """#594 last-input-token c_C, {ctx_id: (Lc,H) fp32}, probe_pool_hash pinned."""
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
    cells in ``main`` (one ``fit_batched_loco_mlp`` call per (n, d_in, p) shape
    group, the #722 vectorize-many-cell-fits mandate; the old per-cell one-group
    call defeated the batching). Returns ``(out, Y_pca)`` so the caller can batch
    the MLP arm over the same PCA target.

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
) -> dict:
    """Batch the MLP validity arm across ALL cells (grouped by (n, d_in, p) shape).

    ``mlp_jobs`` is a list of ``(cell_key, Xc, Y_pca)``. ``fit_batched_loco_mlp``
    requires every group in ONE call to share (n, d_in, p), so cells are bucketed
    by that shape and each bucket is fit in ONE batched call (the #722 mandate —
    the old code called ``fit_batched_loco_mlp([one group])`` per cell, defeating
    the batching entirely). Returns ``{cell_key: mlp_skill}``.
    """
    buckets: dict[tuple, list[MLPGroup]] = {}
    for key, Xc, Y_pca in mlp_jobs:
        shape = (Xc.shape[0], Xc.shape[1], Y_pca.shape[1])
        buckets.setdefault(shape, []).append(MLPGroup(key, Xc, Y_pca))
    y_by_key = {key: Y_pca for key, _Xc, Y_pca in mlp_jobs}
    mlp_skill_by_key: dict = {}
    for shape, groups in buckets.items():
        logger.info("[phase=mlp] batching %d cells at shape %s", len(groups), shape)
        res = fit_batched_loco_mlp(groups, seed=SHUFFLE_NULL_SEED)
        for g in groups:
            mlp = skill_over_mean_r2(res.preds_by_key[g.key], y_by_key[g.key])
            mlp_skill_by_key[g.key] = mlp["skill"]
    return mlp_skill_by_key


def _fit_null_draws(
    Xc: np.ndarray, Yv: np.ndarray, pca_dim: int, n_perms: int, seed: int
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
    return batched_ridge_loco_null_skill(Xc, Y_pca, perm)


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
            a = pos_summaries[c]["turn_nl"][li]
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


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #810 DV (a): reconstruction skill-over-mean R²")
    ap.add_argument(
        "--position-store-hf",
        default="issue658_theory_assumptions/answer_position_sweep",
        help="HF prefix of the Phase B aligned-subset store",
    )
    ap.add_argument("--position-store-dir", default=None, help="local Phase B store (smoke)")
    ap.add_argument(
        "--free-only",
        action="store_true",
        help="Phase A free leg: fit only mean/last/maxp over ALL 50 manifest "
        "contexts (no Phase B position store needed) — runs before Phase B lands",
    )
    ap.add_argument("--out", default=str(PROJECT_ROOT / "eval_results" / "issue_810"))
    ap.add_argument("--summaries", nargs="*", default=None, help="subset of summaries (smoke)")
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="subset of layer indices")
    ap.add_argument("--n-perms", type=int, default=SHUFFLE_NULL_PERMS)
    ap.add_argument("--no-mlp", action="store_true", help="skip the MLP validity arm")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[phase=load] manifest + free summaries + c_C + position store")
    man = load_json(hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset"))
    ctx_ids_all = context_ids_from_manifest(man)
    free_summaries, capture_layers = _load_free_summaries()

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

    cc = _load_cc(ctx_ids, capture_layers)

    default_summaries = ["mean", "last", "maxp"] if args.free_only else summary_names()
    summaries = args.summaries or default_summaries
    layers = args.layers if args.layers is not None else list(range(len(capture_layers)))
    n = len(ctx_ids)
    pca_dim = min(PCA_TARGET_DIM_CAP, max(1, n - 2))
    logger.info(
        "fitting %d summaries x %d layers (n=%d, pca_dim=%d)",
        len(summaries),
        len(layers),
        n,
        pca_dim,
    )

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
            fit, y_pca = _fit_one_cell(Xc, Yv, cell_pca)
            fit["layer"] = capture_layers[li]
            fit["n_kept"] = len(kept)
            cells.append(fit)
            if not args.no_mlp:
                key = (summary, capture_layers[li])
                mlp_jobs.append((key, Xc, y_pca))
                cell_ref[key] = fit
            null_matrix[summary][str(capture_layers[li])] = _fit_null_draws(
                Xc, Yv, cell_pca, args.n_perms, SHUFFLE_NULL_SEED
            )
        results[summary] = cells
        logger.info("[phase=fit] %s done (%d layers)", summary, len(cells))

    # Batch the MLP validity arm across all cells (one call per shape group), then
    # attach mlp_skill back to each cell. Batching (not one-group-per-call) is the
    # #722 fix — the on-main fit_batched_loco_mlp batches (group × dim × fold).
    if mlp_jobs:
        mlp_skill_by_key = _batch_mlp_validity(mlp_jobs)
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

    dump_json(
        {
            "dv": "reconstruction_skill_over_mean_r2",
            "predictor": "cc_last_input_token (#594)",
            "n_contexts": n,
            "pca_target_dim": pca_dim,
            "capture_layers": capture_layers,
            "by_summary": results,
            "diagnostics": diag,
            "reproducibility": reproducibility_metadata(),
            "smoke": args.smoke,
        },
        out_dir / "reconstruction_skill_by_summary.json",
    )
    # Null matrix persisted separately (its own primary-deliverable file; §6.5).
    dump_json(
        {
            "dv": "reconstruction",
            "axes": "summary -> layer -> [per-draw ridge skill]",
            "n_perms": args.n_perms,
            "seed": SHUFFLE_NULL_SEED,
            "reconstruction": null_matrix,
        },
        out_dir / "null_matrix_reconstruction.json",
    )
    logger.info("[phase=done] wrote reconstruction results + null matrix to %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
