#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, ², →) in scientific docstrings + log messages.
"""Issue #810 analyzer calibration: paired-bootstrap Δskill(maxp − mean).

Persists the plan §6 "H1 claim-size calibration" the round-1 analyzer computed
inline (previously unpersisted — interp-critique r1 required the script + draws
summary be committed to ``eval_results/issue_810/analysis/``).

Method
------
For each of {mean, maxp} × 28 layers: rebuild the EXACT reconstruction cell from
``issue810_fit_reconstruction`` (train-fold PCA-48 target, LOCO ridge
predictions via the on-main primitives), then store the PER-CONTEXT
``(ss_res_i, ss_tot_i)`` decomposition of the held-out predictions
(``ss_tot_i`` against the leave-one-out train mean). The paired bootstrap
resamples CONTEXTS with replacement over that fixed decomposition (no
per-replicate refit) — the sampling variability of the skill statistic
``1 − Σss_res/Σss_tot``, computed identically for both summaries on the SAME
resampled context set (paired).

Statistics reported (B draws, one shared RNG stream, seed fixed):

- Δskill at matched layer L18 (mean's best layer).
- Δskill at matched layer L21 (maxp's best layer — a DATA-SELECTED layer;
  labeled as such in the analysis).
- best-layer-vs-best-layer Δskill with the layer selection INHERITED per
  replicate (selection-symmetric), over all layers and over the mid/late
  window L14–22.
- fixed late-window L19–26 window-mean Δskill (per replicate: mean over the
  window of per-layer paired deltas) — the pre-stated support for the
  "maxp is more robust late" claim (fixed window, no data-driven selection).

Usage::

    uv run python scripts/issue810_bootstrap_deltaskill.py \
        --out eval_results/issue_810/analysis/bootstrap_deltaskill.json
"""

from __future__ import annotations

import argparse
import logging

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import numpy as np  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue810_common import (  # noqa: E402
    G1_OUT_DIR,
    HF_DATA_REPO,
    I658_STORE_MANIFEST,
    PCA_TARGET_DIM_CAP,
    context_ids_from_manifest,
    dump_json,
    reproducibility_metadata,
)
from issue810_fit_reconstruction import (  # noqa: E402
    _load_cc,
    _load_cc_for_genre,
    _load_free_summaries,
    _load_v0_blob,
)

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    loco_train_means,
    ridge_predict_loco_centered,
    robust_pca_basis,
)

logger = logging.getLogger("issue810_bootstrap_deltaskill")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SUMMARIES = ("mean", "maxp")
MATCHED_LAYERS = (18, 21)
MIDLATE_WINDOW = tuple(range(14, 23))  # L14–22 (plan §6 mid/late window)
LATE_WINDOW = tuple(range(19, 27))  # L19–26 (fixed window, robustness claim)


def _per_context_decomposition(Xc: np.ndarray, Yv: np.ndarray, pca_dim: int):
    """Per-context (ss_res_i, ss_tot_i) of the held-out LOCO ridge predictions.

    Mirrors ``issue810_fit_reconstruction._fit_one_cell`` exactly (train-fold
    PCA target via robust_pca_basis, LOCO ridge via
    ridge_predict_loco_centered), then decomposes the aggregate skill
    ``1 − Σss_res/Σss_tot`` into its per-context terms.
    Returns (ss_res (n,), ss_tot (n,)).
    """
    mu, comps, _ = robust_pca_basis(Yv, pca_dim)
    Y_pca = (Yv - mu) @ comps.T  # (n, k)
    preds = ridge_predict_loco_centered(Xc, Y_pca)  # (n, k) held-out
    tmean = loco_train_means(Y_pca)  # (n, k) LOO train means
    ss_res = np.sum((Y_pca - preds) ** 2, axis=1)
    ss_tot = np.sum((Y_pca - tmean) ** 2, axis=1)
    return ss_res, ss_tot


def _skill(ss_res: np.ndarray, ss_tot: np.ndarray, idx: np.ndarray | None = None) -> float:
    """Aggregate skill 1 − Σss_res/Σss_tot over (optionally resampled) contexts."""
    if idx is not None:
        ss_res, ss_tot = ss_res[idx], ss_tot[idx]
    tot = float(ss_tot.sum())
    return float("nan") if tot < 1e-12 else 1.0 - float(ss_res.sum()) / tot


def _stat_summary(obs: float, draws: np.ndarray) -> dict:
    """Observed value + bootstrap CI + P(Δ≤0) + draw percentiles."""
    return {
        "observed": obs,
        "ci95": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))],
        "p_delta_le_0": float(np.mean(draws <= 0.0)),
        "draw_percentiles": {
            str(p): float(np.percentile(draws, p)) for p in (2.5, 25, 50, 75, 97.5)
        },
        "n_draws": int(draws.size),
    }


# ── uh new-rows paired Δskill vs the mean benchmark (plan v11 §4.5 step 6) ────


def _vs_mean_rows(args) -> int:
    """Paired per-context bootstrap of Δskill(new_row − mean) at the selected cells.

    The `user-header-newline-summary` round's claim-size calibration: for each
    of the 9 new rows (from the ``--uh-summaries`` pack) vs the `mean`
    benchmark (v0), per-context (ss_res, ss_tot) decompositions are computed in
    the SAME canonical context order and every bootstrap replicate resamples
    the SAME context indices for both rows (paired; ONE shared (B, n) index
    matrix — no per-draw refit loop, the #722 vectorize mandate). Statistics
    per row: Δ at the mean's benchmark L18 (frozen), best-vs-best with the
    layer selection INHERITED per replicate, and Δ frozen at each row's
    observed best layer (data-selected, labeled). Output: ``delta_vs_mean.json``.
    """
    import json

    from huggingface_hub import hf_hub_download
    from issue810_common import UH_SUMMARY_NAMES, validate_uh_pack
    from issue810_fit_readout import _load_uh_summaries

    man_path = hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset")
    with open(man_path) as f:
        ctx_ids = context_ids_from_manifest(json.load(f))
    n = len(ctx_ids)
    pca_dim = min(PCA_TARGET_DIM_CAP, n - 2)
    if not args.uh_summaries:
        raise SystemExit("--vs mean requires --uh-summaries (the new-row source pack)")
    uh_rows, uh_cov, meta = _load_uh_summaries(args.uh_summaries)
    rows = args.rows or list(UH_SUMMARY_NAMES)
    unknown = [r for r in rows if r not in uh_rows]
    if unknown:
        raise SystemExit(f"rows {unknown} absent from the uh_summaries pack ({sorted(uh_rows)})")
    free_summaries, capture_layers = _load_free_summaries()
    if not meta.get("smoke"):
        # PRODUCTION pack: ALL requested rows must carry a full-layer tensor +
        # positive coverage for EVERY manifest context, on the production model
        # (r1 CONCERN uh-pack-validation-bootstrap — rows[0]-only coverage +
        # min()-layer truncation let a partial/truncated pack through silently).
        # Raises UhPackValidationError BEFORE any decomposition/output.
        validate_uh_pack(
            uh_rows,
            uh_cov,
            meta,
            requested_rows=rows,
            ctx_ids=ctx_ids,
            expected_capture_layers=capture_layers,
        )
        layers = args.layers or list(range(len(capture_layers)))
    else:
        # SMOKE-provenance pack (tiny ctx subset / non-7B layer count): pair on
        # the subset covered by ALL requested rows, loudly; the layer window
        # truncates to the pack's own axis. This relaxed path is smoke-ONLY.
        covered = [c for c in ctx_ids if all(uh_cov[r].get(c, 0) > 0 for r in rows)]
        if len(covered) < n:
            ctx_ids = covered
            n = len(ctx_ids)
            if n < 8:
                raise SystemExit(f"smoke pack covers only {n} contexts (<8) — too small to fit")
            pca_dim = min(PCA_TARGET_DIM_CAP, n - 2)
            logger.warning(
                "[vs-mean] SMOKE pack: pairing on its %d covered contexts (production covers 50)",
                n,
            )
        n_layers_pack = next(iter(uh_rows[rows[0]].values())).shape[0]
        layers = args.layers or list(range(min(len(capture_layers), n_layers_pack)))
    cc = _load_cc(ctx_ids, capture_layers)
    n_layers = len(layers)
    logger.info("[vs-mean] rows=%s n=%d layers=%d pca_dim=%d", rows, n, n_layers, pca_dim)

    arms = ["mean", *rows]
    ss_res = {s: np.zeros((n_layers, n)) for s in arms}
    ss_tot = {s: np.zeros((n_layers, n)) for s in arms}
    obs_skill = {s: np.zeros(n_layers) for s in arms}
    for s in arms:
        for wi, li in enumerate(layers):
            Xc = np.stack([cc[c][li] for c in ctx_ids])
            if s == "mean":
                Yv = np.stack([free_summaries["mean"][c][li].numpy() for c in ctx_ids])
            else:
                Yv = np.stack([uh_rows[s][c][li] for c in ctx_ids])
            r, t = _per_context_decomposition(Xc, Yv, pca_dim)
            ss_res[s][wi], ss_tot[s][wi] = r, t
            obs_skill[s][wi] = _skill(r, t)
        logger.info(
            "[vs-mean] %s best skill %.4f @window-idx %d",
            s,
            float(obs_skill[s].max()),
            int(obs_skill[s].argmax()),
        )

    rng = np.random.default_rng(args.seed)
    B = args.n_boot
    idx = rng.integers(0, n, size=(B, n))  # ONE index matrix — paired across arms
    skills = {}
    for s in arms:
        rs = ss_res[s][:, idx].sum(axis=-1)  # (n_layers, B)
        ts = ss_tot[s][:, idx].sum(axis=-1)
        skills[s] = np.where(ts < 1e-12, np.nan, 1.0 - rs / ts)

    layer_pos = {capture_layers[li]: wi for wi, li in enumerate(layers)}
    statistics: dict[str, dict] = {}
    for s in rows:
        d_draws = skills[s] - skills["mean"]  # (n_layers, B)
        d_obs = obs_skill[s] - obs_skill["mean"]
        if 18 in layer_pos:  # the mean benchmark's layer (frozen, registered anchor)
            wi = layer_pos[18]
            statistics[f"{s}_at_L18"] = _stat_summary(float(d_obs[wi]), d_draws[wi])
        draws_inh = skills[s].max(axis=0) - skills["mean"].max(axis=0)
        obs_inh = float(obs_skill[s].max() - obs_skill["mean"].max())
        statistics[f"{s}_best_vs_best_inherited"] = _stat_summary(obs_inh, draws_inh)
        bi = int(obs_skill[s].argmax())
        st = _stat_summary(float(d_obs[bi]), d_draws[bi])
        st["row_best_layer"] = capture_layers[layers[bi]]
        st["note"] = "row's own best layer (DATA-SELECTED; not multiplicity-corrected)"
        statistics[f"{s}_frozen_observed_best_layer"] = st

    out_path = args.out or str(
        PROJECT_ROOT
        / "eval_results"
        / "issue_810"
        / "user-header-newline-summary"
        / "delta_vs_mean.json"
    )
    dump_json(
        {
            "dv": "paired_bootstrap_delta_skill_new_row_minus_mean",
            "method": (
                "per-context (ss_res, ss_tot) decompositions per (row, layer) in the SAME "
                "canonical context order; ONE shared (B, n) context-resample index matrix "
                "(paired across the mean benchmark + every new row); Δ at L18 (frozen), "
                "best-vs-best inherited per replicate, and each row's observed-best layer "
                "(data-selected, labeled)"
            ),
            "rows": rows,
            "n_contexts": n,
            "layers_window_indices": list(layers),
            "pca_dim": pca_dim,
            "n_boot": B,
            "seed": args.seed,
            "per_layer_observed_skill": {s: [float(v) for v in obs_skill[s]] for s in arms},
            "statistics": statistics,
            "reproducibility": reproducibility_metadata(),
        },
        out_path,
    )
    for k, v in statistics.items():
        logger.info(
            "%s: obs %+0.4f CI95 [%+0.4f, %+0.4f] P(<=0)=%.4f",
            k,
            v["observed"],
            v["ci95"][0],
            v["ci95"][1],
            v["p_delta_le_0"],
        )
    logger.info("wrote %s", out_path)
    return 0


# ── cross-genre paired Δskill (plan v6 H1-g(ii); follow-up round) ──────────────

GENRE_GENERALITY_BAND = 0.10  # plan v6 §11: parent's largest within-genre summary effect


def _cross_genre(args) -> int:
    """Paired per-context bootstrap of Δskill = skill_Betley − skill_g1 (H1-g(ii)).

    Contexts are IDENTICAL across arms (plan-verified), so every read is PAIRED
    per context: both genres' per-context (ss_res, ss_tot) decompositions are
    indexed by the SAME canonical ctx list (Betley manifest order) and every
    bootstrap replicate resamples the SAME context indices for both arms.
    REGISTERED read (H1-g(ii)): the `mean` summary at each arm's best layer with
    the layer selection INHERITED per replicate; CONFIRMS transfer when the 95%
    CI upper bound <= 0.10 (the genre-generality band — the parent's own largest
    within-genre summary effect). Frozen-cell companions (mean@L18, maxp@L21,
    observed-best frozen) are reported alongside, never the decider. All draws
    are vectorized (one (B, n) index matrix; per-cell resampled sums as one
    fancy-index reduction — no per-draw Python refit loop).
    """
    from huggingface_hub import hf_hub_download

    man_path = hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset")
    import json

    with open(man_path) as f:
        ctx_ids = context_ids_from_manifest(json.load(f))
    n = len(ctx_ids)
    pca_dim = min(PCA_TARGET_DIM_CAP, n - 2)

    genres = ("betley", "g1")
    free, cc = {}, {}
    for g in genres:
        free[g], capture_layers = _load_free_summaries(g)
        cc[g] = _load_cc_for_genre(g, ctx_ids, capture_layers)
    # Context-set identity across arms (paired design precondition, plan §5).
    g1_ctx = set(_load_v0_blob("g1")["context_ids"])
    if g1_ctx != set(ctx_ids):
        raise RuntimeError(
            f"g1 store context set differs from Betley's: {sorted(g1_ctx ^ set(ctx_ids))[:5]}"
        )
    layers = args.layers if args.layers is not None else list(range(len(capture_layers)))
    n_layers = len(layers)
    logger.info(
        "[cross-genre] n=%d layers=%d pca_dim=%d (decompositions: 2 genres x 2 summaries)",
        n,
        n_layers,
        pca_dim,
    )

    # Per-(genre, summary, layer) per-context decompositions, canonical ctx order.
    ss_res = {(g, s): np.zeros((n_layers, n)) for g in genres for s in SUMMARIES}
    ss_tot = {(g, s): np.zeros((n_layers, n)) for g in genres for s in SUMMARIES}
    obs_skill = {(g, s): np.zeros(n_layers) for g in genres for s in SUMMARIES}
    import time as _time

    cell_walls: list[float] = []
    for g in genres:
        for s in SUMMARIES:
            for wi, li in enumerate(layers):
                t0 = _time.time()
                Xc = np.stack([cc[g][c][li] for c in ctx_ids])
                Yv = np.stack([free[g][s][c][li].numpy() for c in ctx_ids])
                r, t = _per_context_decomposition(Xc, Yv, pca_dim)
                cell_walls.append(_time.time() - t0)
                ss_res[(g, s)][wi], ss_tot[(g, s)][wi] = r, t
                obs_skill[(g, s)][wi] = _skill(r, t)
            logger.info(
                "[cross-genre] %s/%s best skill %.4f @window-idx %d (%.2fs/cell mean)",
                g,
                s,
                float(obs_skill[(g, s)].max()),
                int(obs_skill[(g, s)].argmax()),
                float(np.mean(cell_walls)),
            )

    rng = np.random.default_rng(args.seed)
    B = args.n_boot
    idx = rng.integers(0, n, size=(B, n))  # ONE index matrix — paired across arms
    # skills[(g, s)] (n_layers, B): resampled sums via one fancy-index reduction.
    skills = {}
    for key in ss_res:
        rs = ss_res[key][:, idx].sum(axis=-1)  # (n_layers, B)
        ts = ss_tot[key][:, idx].sum(axis=-1)
        skills[key] = np.where(ts < 1e-12, np.nan, 1.0 - rs / ts)

    layer_pos = {layer: wi for wi, layer in enumerate(layers)}
    statistics: dict[str, dict] = {}
    for s in SUMMARIES:
        d_draws_by_layer = skills[("betley", s)] - skills[("g1", s)]  # (n_layers, B)
        d_obs_by_layer = obs_skill[("betley", s)] - obs_skill[("g1", s)]
        # REGISTERED (mean summary): best-vs-best with selection INHERITED per draw.
        draws_inh = skills[("betley", s)].max(axis=0) - skills[("g1", s)].max(axis=0)
        obs_inh = float(obs_skill[("betley", s)].max() - obs_skill[("g1", s)].max())
        statistics[f"{s}_best_vs_best_inherited"] = _stat_summary(obs_inh, draws_inh)
        # Frozen at each arm's OBSERVED best layer (selection frozen, companion).
        bi_b = int(obs_skill[("betley", s)].argmax())
        bi_g = int(obs_skill[("g1", s)].argmax())
        draws_frozen = skills[("betley", s)][bi_b] - skills[("g1", s)][bi_g]
        obs_frozen = float(obs_skill[("betley", s)][bi_b] - obs_skill[("g1", s)][bi_g])
        st = _stat_summary(obs_frozen, draws_frozen)
        st["betley_best_layer"] = layers[bi_b]
        st["g1_best_layer"] = layers[bi_g]
        statistics[f"{s}_frozen_observed_best_layers"] = st
        # Parent-selected frozen cells (mean@L18 / maxp@L21), when in the subset.
        frozen_layer = 18 if s == "mean" else 21
        if frozen_layer in layer_pos:
            wi = layer_pos[frozen_layer]
            statistics[f"{s}_frozen_L{frozen_layer}"] = _stat_summary(
                float(d_obs_by_layer[wi]), d_draws_by_layer[wi]
            )

    reg = statistics["mean_best_vs_best_inherited"]
    reg["registered_read"] = True
    reg["genre_generality_band"] = GENRE_GENERALITY_BAND
    reg["ci95_upper_within_band"] = bool(reg["ci95"][1] <= GENRE_GENERALITY_BAND)
    reg["ci95_lower_exceeds_band"] = bool(reg["ci95"][0] > GENRE_GENERALITY_BAND)

    out_path = args.out or str(G1_OUT_DIR / "genre_delta_recon.json")
    dump_json(
        {
            "dv": "paired_cross_genre_delta_skill_betley_minus_g1",
            "method": (
                "per-context (ss_res, ss_tot) decompositions computed per (genre, summary, "
                "layer) in the SAME canonical Betley-manifest context order for both arms; "
                "one shared (B, n) context-resample index matrix (paired across arms + "
                "summaries); registered H1-g(ii) read = mean summary, best layer inherited "
                "per replicate, 95% CI upper bound vs the 0.10 genre-generality band"
            ),
            "n_contexts": n,
            "layers": list(layers),
            "pca_dim": pca_dim,
            "n_boot": B,
            "seed": args.seed,
            "mean_cell_wall_s": float(np.mean(cell_walls)),
            "per_layer_observed_skill": {
                f"{g}/{s}": [float(v) for v in obs_skill[(g, s)]] for g in genres for s in SUMMARIES
            },
            "statistics": statistics,
            "reproducibility": reproducibility_metadata(),
        },
        out_path,
    )
    for k, v in statistics.items():
        logger.info(
            "%s: obs %+0.4f CI95 [%+0.4f, %+0.4f] P(<=0)=%.4f",
            k,
            v["observed"],
            v["ci95"][0],
            v["ci95"][1],
            v["p_delta_le_0"],
        )
    logger.info("wrote %s", out_path)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #810 paired-bootstrap Δskill(maxp − mean)")
    ap.add_argument(
        "--out",
        default=None,
        help="output JSON (default: eval_results/issue_810/analysis/bootstrap_deltaskill.json; "
        "with --cross-genre: eval_results/issue_810/ultrachat-genre-summary-sweep/"
        "genre_delta_recon.json)",
    )
    ap.add_argument("--n-boot", "--n-draws", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--cross-genre",
        action="store_true",
        help="paired Betley-vs-g1 Δskill (plan v6 H1-g(ii)) instead of the within-genre "
        "maxp-vs-mean delta",
    )
    ap.add_argument(
        "--vs",
        choices=["mean"],
        default=None,
        help="'mean' = the uh round's paired Δskill(new_row − mean benchmark) mode "
        "(plan v11 §4.5 step 6; requires --uh-summaries; writes delta_vs_mean.json)",
    )
    ap.add_argument(
        "--rows",
        nargs="*",
        default=None,
        help="new-row subset for --vs mean (default: all 9 uh rows in the pack)",
    )
    ap.add_argument(
        "--uh-summaries",
        default=None,
        help="uh_summaries.pt pack (local path or HF data-repo path) — the new-row source",
    )
    ap.add_argument(
        "--layers",
        nargs="*",
        type=int,
        default=None,
        help="layer-index subset (smoke); cross-genre only — the within-genre path always "
        "runs all 28 layers (parent behavior, bit-for-bit)",
    )
    args = ap.parse_args()
    if args.vs == "mean":
        return _vs_mean_rows(args)
    if args.cross_genre:
        return _cross_genre(args)
    if args.out is None:
        args.out = str(
            PROJECT_ROOT / "eval_results" / "issue_810" / "analysis" / "bootstrap_deltaskill.json"
        )

    from huggingface_hub import hf_hub_download

    man_path = hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset")
    import json

    with open(man_path) as f:
        ctx_ids = context_ids_from_manifest(json.load(f))
    free_summaries, capture_layers = _load_free_summaries()
    cc = _load_cc(ctx_ids, capture_layers)
    n = len(ctx_ids)
    n_layers = len(capture_layers)
    pca_dim = min(PCA_TARGET_DIM_CAP, n - 2)
    logger.info("n_contexts=%d n_layers=%d pca_dim=%d", n, n_layers, pca_dim)

    # Per-cell per-context decompositions: {summary: (n_layers, n) arrays}.
    ss_res = {s: np.zeros((n_layers, n)) for s in SUMMARIES}
    ss_tot = {s: np.zeros((n_layers, n)) for s in SUMMARIES}
    obs_skill = {s: np.zeros(n_layers) for s in SUMMARIES}
    for s in SUMMARIES:
        for li in range(n_layers):
            Xc = np.stack([cc[c][li] for c in ctx_ids])
            Yv = np.stack([free_summaries[s][c][li].numpy() for c in ctx_ids])
            r, t = _per_context_decomposition(Xc, Yv, pca_dim)
            ss_res[s][li], ss_tot[s][li] = r, t
            obs_skill[s][li] = _skill(r, t)
        logger.info(
            "[%s] best skill %.4f @L%d",
            s,
            float(obs_skill[s].max()),
            int(obs_skill[s].argmax()),
        )

    rng = np.random.default_rng(args.seed)
    B = args.n_boot
    draws = {
        "matched_L18": np.zeros(B),
        "matched_L21": np.zeros(B),
        "best_vs_best_all": np.zeros(B),
        "best_vs_best_midlate_L14_22": np.zeros(B),
        "window_mean_L19_26": np.zeros(B),
    }
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        sk = {
            s: np.array([_skill(ss_res[s][li], ss_tot[s][li], idx) for li in range(n_layers)])
            for s in SUMMARIES
        }
        d = sk["maxp"] - sk["mean"]
        draws["matched_L18"][b] = d[18]
        draws["matched_L21"][b] = d[21]
        draws["best_vs_best_all"][b] = sk["maxp"].max() - sk["mean"].max()
        ml = list(MIDLATE_WINDOW)
        draws["best_vs_best_midlate_L14_22"][b] = sk["maxp"][ml].max() - sk["mean"][ml].max()
        draws["window_mean_L19_26"][b] = float(np.mean(d[list(LATE_WINDOW)]))

    d_obs = obs_skill["maxp"] - obs_skill["mean"]
    ml = list(MIDLATE_WINDOW)
    observed = {
        "matched_L18": float(d_obs[18]),
        "matched_L21": float(d_obs[21]),
        "best_vs_best_all": float(obs_skill["maxp"].max() - obs_skill["mean"].max()),
        "best_vs_best_midlate_L14_22": float(
            obs_skill["maxp"][ml].max() - obs_skill["mean"][ml].max()
        ),
        "window_mean_L19_26": float(np.mean(d_obs[list(LATE_WINDOW)])),
    }

    out = {
        "dv": "paired_bootstrap_delta_skill_maxp_minus_mean",
        "method": (
            "contexts resampled with replacement over the fixed per-context "
            "(ss_res, ss_tot) decomposition of the held-out LOCO ridge predictions "
            "(train-fold PCA-48 target; no per-replicate refit); layer selection for "
            "best-vs-best statistics inherited per replicate; L19-26 window mean is a "
            "fixed (non-selected) window"
        ),
        "n_contexts": n,
        "pca_dim": pca_dim,
        "n_boot": B,
        "seed": args.seed,
        "note_L21_is_data_selected": (
            "L21 is maxp's own best layer (data-selected); the matched_L21 CI is "
            "conditional on that selection and is NOT multiplicity-corrected"
        ),
        "per_layer_observed_skill": {s: [float(v) for v in obs_skill[s]] for s in SUMMARIES},
        "per_context_decomposition": {
            s: {
                "ss_res": ss_res[s].tolist(),
                "ss_tot": ss_tot[s].tolist(),
                "context_ids": ctx_ids,
            }
            for s in SUMMARIES
        },
        "statistics": {k: _stat_summary(observed[k], draws[k]) for k in draws},
        "reproducibility": reproducibility_metadata(),
    }
    dump_json(out, args.out)
    for k, v in out["statistics"].items():
        logger.info(
            "%s: obs %+0.4f CI95 [%+0.4f, %+0.4f] P(<=0)=%.4f",
            k,
            v["observed"],
            v["ci95"][0],
            v["ci95"][1],
            v["p_delta_le_0"],
        )
    logger.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
