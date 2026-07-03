#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
# Intentional Unicode (Δ, ², ≥, →) in scientific docstrings + log messages.
"""Issue #810 `_he` mechanism + paired-equivalence driver (plan v15 §4.6 item 4).

The header-echo ablation round's CPU-chain analysis, a thin driver over the
existing batched primitives (the ``issue810_uh_crosslayer.py`` precedent —
no new fit machinery):

(a) **Mechanism read (H2-he).** Cross-context-centers the full-answer and
    empty-answer header activations per (row × layer) over the SAME 50
    contexts and emits per-context centered cosine + pooled R²
    (1 − ‖Fc−Ec‖²/‖Fc‖², centered), PLUS the per-(row × layer) EMPTY-side
    centered-norm/variance summary — the degeneracy key (plan v15 §8 risk 1:
    centered cosine ≈ 1 is the ECHO signature; collapse shows as LOW
    empty-side variance, cosine → ~0/undefined).
(b) **Full-side refit parity.** Recomputes each row's full-answer LOCO skill
    at its COMMITTED best layer and asserts it matches the committed value
    ≤ 1e-6 (the ``a26da411bb`` precedent): the 7 uh/bnd rows vs round-3
    ``reconstruction_skill_user_header.json`` (Y from ``uh_summaries.pt``);
    ``im_end``/``turn_nl`` vs the ROUND-1 committed
    ``reconstruction_skill_by_summary.json``, Y from the ROUND-1 HF store
    per-file (``answer_position_sweep/`` — the exact data those committed
    skills were fit from). NOTE (implementation deviation from plan v15 §4.6
    item 4, which named the round-3 store on a byte-identity premise): the
    round-3 recapture of these two positions byte-differs from round-1 on
    5/50 contexts (bf16 batched-recapture noise; the round-3 drift check
    covered ctx0 only), putting the round-3-sourced refit at |Δ|≈3.8e-5 —
    the round-1 store satisfies the REGISTERED ≤1e-6 parity (measured
    |Δ|≈6e-8) and is the committed comparator's own data. FAIL ⇒ halt
    (``failure_class: data``).
(c) **Paired shared-index bootstrap (H1-he PRIMARY).** Δskill(full − empty)
    per row at the committed best layers, 2,000 draws seed 42, ONE shared
    (B, n) context-resample index matrix across ALL rows and both sides (no
    per-draw refit — the fixed per-context (ss_res, ss_tot) decomposition,
    the ``issue810_bootstrap_deltaskill`` machinery). Per row: ci95,
    contained_within_margin (±0.02, the equivalence read), verdict ∈
    {echo_consistent, failure_to_reject, positive_gap}. Familywise:
    P(≥1 |Δ̄_centered| ≥ 0.02 | null) formed by centering the shared-index
    joint draws per row (preserves cross-row dependence); draws + the read
    PERSIST inside ``paired_full_minus_empty.json``.

ROW-COVERAGE ASSERT at driver start (plan v15 Must-Fix, mechanizable):
``set(registered_pair_rows) ⊆ union(keys(full_side_sources))``.

Usage::

    # production (cpu-mid, after the GPU phase lands he_summaries.pt on HF):
    uv run python scripts/issue810_he_mechanism.py \\
        --he-summaries data/issue_810/he_summaries.pt \\
        --out eval_results/issue_810/header-echo-ablation-capture

    # smoke (2 rows incl. one round-1-targeted; self-pairs the empty side
    # when no production he pack exists yet — refit parity still real):
    uv run python scripts/issue810_he_mechanism.py --smoke --n-boot 200 \\
        --out /tmp/i810_he_smoke/mech
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

import torch  # noqa: E402

# Cross-script helper imports hoisted to module top so a missing symbol crashes
# at process start, never inside a smoke-skipped branch (gotchas.md #606).
from issue810_bootstrap_deltaskill import (  # noqa: E402
    _per_context_decomposition,
    _skill,
    _stat_summary,
)
from issue810_common import (  # noqa: E402
    ANSWER_POSITION_SWEEP_SUBDIR,
    HE_SUMMARY_NAMES,
    HF_DATA_REPO,
    HF_PREFIX,
    I658_STORE_MANIFEST,
    PCA_TARGET_DIM_CAP,
    UH_SUMMARIES_HF_FILE,
    context_ids_from_manifest,
    dump_json,
    load_json,
    reproducibility_metadata,
    validate_uh_pack,
)
from issue810_fit_readout import _load_uh_summaries  # noqa: E402
from issue810_fit_reconstruction import (  # noqa: E402
    _committed_best_layer,
    _load_cc,
    _load_free_summaries,
)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue810_he_mechanism")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# The registered H1-he pairing (plan v15 §3): 9 pairs, split by full-side source.
UH_PACK_FULL_ROWS: tuple[str, ...] = (
    "uh_im_start",
    "uh_user",
    "uh_nl",
    "uh_mean3",
    "uh_max3",
    "bnd_mean5",
    "bnd_max5",
)
R1_STORE_FULL_ROWS: tuple[str, ...] = ("im_end", "turn_nl")  # full side: ROUND-1 store per-file
EQUIVALENCE_MARGIN = 0.02  # the round-3 D_uh margin as a non-inferiority bound (plan v15 §3)
REFIT_PARITY_TOL = 1e-6  # the a26da411bb precedent (plan v15 kill criterion 3)


def _assert_row_coverage(rows: list[str]) -> None:
    """Plan v15 Must-Fix: every registered pair row has a NAMED full-side source."""
    known = set(UH_PACK_FULL_ROWS) | set(R1_STORE_FULL_ROWS)
    orphans = sorted(set(rows) - known)
    if orphans:
        raise RuntimeError(
            f"row-coverage assert FAILED: rows {orphans} have no registered full-side "
            f"source (uh pack: {sorted(UH_PACK_FULL_ROWS)}; round-1 store: "
            f"{sorted(R1_STORE_FULL_ROWS)})"
        )


def _load_r1_store_rows(ctx_ids: list[str], rows: list[str]) -> dict[str, dict[str, np.ndarray]]:
    """{row: {ctx: (Lc, H) fp32 np}} from the ROUND-1 position store, per-file.

    Per-file ``hf_hub_download`` (~7 MB/ctx, ~350 MB total — cpu-mid-safe);
    only the requested rows are retained in memory. The round-1 store is the
    im_end/turn_nl full side BECAUSE its committed skills are the refit-parity
    target (see the module docstring's deviation note: the round-3 recapture
    byte-differs on 5/50 contexts and misses the registered ≤1e-6 parity).
    Fails loud on a missing row / zero coverage (never a silent skip).
    """
    from huggingface_hub import hf_hub_download

    out: dict[str, dict[str, np.ndarray]] = {r: {} for r in rows}
    for c in ctx_ids:
        p = hf_hub_download(
            HF_DATA_REPO,
            f"{HF_PREFIX}/{ANSWER_POSITION_SWEEP_SUBDIR}/{c}.pt",
            repo_type="dataset",
        )
        blob = torch.load(p, weights_only=False)
        pos = {name: i for i, name in enumerate(blob["positions"])}
        for r in rows:
            if r not in pos or blob["coverage"].get(r, 0) <= 0:
                raise RuntimeError(f"round-1 store {c}: row {r!r} absent or zero-coverage")
            out[r][c] = blob["pos_vectors"][pos[r]].float().numpy()
    return out


def _full_side_sources(rows: list[str], ctx_ids: list[str], capture_layers: list[int], args):
    """Load the full-answer side per row + the committed refit targets.

    Returns ``(full, committed_ref)`` where ``full[row][ctx] = (Lc, H) fp32``
    and ``committed_ref[row] = (best_layer, committed_skill, source_tag)``.
    The uh pack is production-validated (``validate_uh_pack``); the
    im_end/turn_nl full side reads the ROUND-1 store per-file (see
    ``_load_r1_store_rows`` — the committed refit-parity target's own data).
    """
    full: dict[str, dict[str, np.ndarray]] = {}
    committed_ref: dict[str, tuple[int, float, str]] = {}
    pack_rows = [r for r in rows if r in UH_PACK_FULL_ROWS]
    store_rows = [r for r in rows if r in R1_STORE_FULL_ROWS]
    if pack_rows:
        uh_rows, uh_cov, uh_meta = _load_uh_summaries(args.uh_summaries)
        if uh_meta.get("ablate_answer"):
            raise RuntimeError("--uh-summaries points at an ABLATED pack — not the full side")
        validate_uh_pack(
            {r: {c: torch.from_numpy(v) for c, v in per.items()} for r, per in uh_rows.items()},
            uh_cov,
            uh_meta,
            requested_rows=pack_rows,
            ctx_ids=ctx_ids,
            expected_capture_layers=capture_layers,
        )
        r3_points = load_json(args.committed_recon_uh)["by_summary"]
        for r in pack_rows:
            full[r] = uh_rows[r]
            layer = _committed_best_layer(args.committed_recon_uh, r)
            ref = next(c for c in r3_points[r] if int(c["layer"]) == layer)
            committed_ref[r] = (layer, float(ref["ridge_skill"]), "round3")
    if store_rows:
        store = _load_r1_store_rows(ctx_ids, store_rows)
        r1_points = load_json(args.committed_recon)["by_summary"]
        for r in store_rows:
            full[r] = store[r]
            layer = _committed_best_layer(args.committed_recon, r)
            ref = next(c for c in r1_points[r] if int(c["layer"]) == layer)
            committed_ref[r] = (layer, float(ref["ridge_skill"]), "round1")
    missing = sorted(set(rows) - set(full))
    if missing:
        raise RuntimeError(f"full side missing rows {missing} after source load")
    return full, committed_ref


def _load_empty_side(args, rows: list[str], ctx_ids: list[str], capture_layers: list[int]):
    """The empty-answer side from ``he_summaries.pt`` (production-validated).

    Smoke fallback (--smoke only): a missing / smoke-provenance / shape-
    mismatched pack SELF-PAIRS the empty side to the full side (Δ ≡ 0 — the
    machinery, asserts, refit parity, and JSON contracts are still exercised
    end-to-end on real full-side data), LOUDLY logged, never silent.
    """
    if args.he_summaries:
        he_rows, he_cov, he_meta = _load_uh_summaries(args.he_summaries)
        if not args.smoke:
            if he_meta.get("ablate_answer") is not True:
                raise RuntimeError(
                    "--he-summaries pack lacks ablate_answer=True provenance — refusing "
                    "(this driver pairs full-answer vs EMPTY-answer captures)"
                )
            validate_uh_pack(
                {r: {c: torch.from_numpy(v) for c, v in per.items()} for r, per in he_rows.items()},
                he_cov,
                he_meta,
                requested_rows=rows,
                ctx_ids=ctx_ids,
                expected_capture_layers=capture_layers,
            )
            return {r: he_rows[r] for r in rows}, he_meta
        # --smoke with a pack: usable only if every (row, ctx) matches shape.
        usable = all(
            r in he_rows
            and all(c in he_rows[r] for c in ctx_ids)
            and he_rows[r][ctx_ids[0]].shape[0] == len(capture_layers)
            for r in rows
        )
        if usable:
            logger.info("[smoke] pairing against the provided he pack (%s)", args.he_summaries)
            return {r: he_rows[r] for r in rows}, he_meta
        logger.warning(
            "[smoke] he pack at %s is smoke-provenance/shape-mismatched (model=%s, "
            "smoke=%s) — SELF-PAIRING the empty side to the full side (Δ ≡ 0); the "
            "loader/schema path was still exercised on the real pack file",
            args.he_summaries,
            he_meta.get("model"),
            he_meta.get("smoke"),
        )
        return None, he_meta
    if not args.smoke:
        raise SystemExit("--he-summaries is REQUIRED outside --smoke (the empty-answer side)")
    logger.warning("[smoke] no --he-summaries — SELF-PAIRING the empty side (Δ ≡ 0)")
    return None, {"self_paired": True}


def _mechanism_read(
    full: dict,
    empty: dict,
    rows: list[str],
    ctx_ids: list[str],
    capture_layers: list[int],
    committed_ref: dict,
) -> dict:
    """(a) Cross-context-centered cosine + pooled R² + empty-side degeneracy key.

    Vectorized over (ctx × layer): per row, both sides are stacked to
    (n, Lc, H) fp64, centered across contexts, then reduced with einsum —
    no per-cell Python fit loop (the standing vectorize mandate).
    """
    out: dict[str, dict] = {}
    for r in rows:
        F = np.stack([full[r][c] for c in ctx_ids]).astype(np.float64)  # (n, Lc, H)
        E = np.stack([empty[r][c] for c in ctx_ids]).astype(np.float64)
        Fc = F - F.mean(axis=0, keepdims=True)
        Ec = E - E.mean(axis=0, keepdims=True)
        dot = np.einsum("nlh,nlh->nl", Fc, Ec)
        nF = np.sqrt(np.einsum("nlh,nlh->nl", Fc, Fc))
        nE = np.sqrt(np.einsum("nlh,nlh->nl", Ec, Ec))
        denom = np.maximum(nF * nE, 1e-12)
        cos = dot / denom  # (n, Lc) per-context centered cosine
        ss_ff = np.einsum("nlh,nlh->l", Fc, Fc)
        ss_diff = np.einsum("nlh,nlh->l", Fc - Ec, Fc - Ec)
        pooled_r2 = 1.0 - ss_diff / np.maximum(ss_ff, 1e-12)
        ss_ee = np.einsum("nlh,nlh->l", Ec, Ec)
        best_layer = committed_ref[r][0]
        per_layer = {}
        for wi, layer in enumerate(capture_layers):
            per_layer[str(layer)] = {
                "mean_centered_cos": float(cos[:, wi].mean()),
                "median_centered_cos": float(np.median(cos[:, wi])),
                "std_centered_cos": float(cos[:, wi].std()),
                "pooled_r2": float(pooled_r2[wi]),
                # Empty-side degeneracy key (plan v15 §8 risk 1): collapse shows
                # as LOW empty-side centered variance/norms, NOT as cosine ≈ 1.
                "empty_centered_norm_mean": float(nE[:, wi].mean()),
                "empty_centered_norm_median": float(np.median(nE[:, wi])),
                "empty_total_centered_var": float(ss_ee[wi] / len(ctx_ids)),
                "full_total_centered_var": float(ss_ff[wi] / len(ctx_ids)),
                "var_ratio_empty_over_full": float(ss_ee[wi] / np.maximum(ss_ff[wi], 1e-12)),
            }
        bi = capture_layers.index(best_layer)
        out[r] = {
            "committed_best_layer": best_layer,
            "per_layer": per_layer,
            "per_context_centered_cos_at_best_layer": {
                c: float(cos[i, bi]) for i, c in enumerate(ctx_ids)
            },
        }
        logger.info(
            "[mechanism] %s @L%d median centered cos %.4f pooled R² %.4f var-ratio %.4f",
            r,
            best_layer,
            out[r]["per_layer"][str(best_layer)]["median_centered_cos"],
            out[r]["per_layer"][str(best_layer)]["pooled_r2"],
            out[r]["per_layer"][str(best_layer)]["var_ratio_empty_over_full"],
        )
    return out


def _verdict(ci_lo: float, ci_hi: float, margin: float = EQUIVALENCE_MARGIN) -> str:
    """Plan v15 §3 H1-he per-row verdict (equivalence-first, then direction)."""
    if -margin < ci_lo and ci_hi < margin:
        return "echo_consistent"
    if ci_lo > 0:
        return "positive_gap"
    return "failure_to_reject"


def _paired_bootstrap(args, full, empty, rows, ctx_ids, capture_layers, committed_ref, cc) -> dict:
    """(b)+(c): full-side refit parity + the shared-index paired bootstrap."""
    n = len(ctx_ids)
    pca_dim = min(PCA_TARGET_DIM_CAP, n - 2)
    ss = {}  # (row, side) -> (ss_res (n,), ss_tot (n,))
    obs_skill = {}
    refit_parity: dict[str, dict] = {}
    for r in rows:
        layer, committed_skill, source = committed_ref[r]
        li = capture_layers.index(layer)
        Xc = np.stack([cc[c][li] for c in ctx_ids])
        for side, src in (("full", full), ("empty", empty)):
            Yv = np.stack([src[r][c][li] for c in ctx_ids])
            res, tot = _per_context_decomposition(Xc, Yv, pca_dim)
            ss[(r, side)] = (res, tot)
            obs_skill[(r, side)] = _skill(res, tot)
        diff = abs(obs_skill[(r, "full")] - committed_skill)
        refit_parity[r] = {
            "committed_best_layer": layer,
            "source": source,
            "committed_skill": committed_skill,
            "recomputed_skill": obs_skill[(r, "full")],
            "abs_diff": diff,
        }
        if diff > REFIT_PARITY_TOL:
            raise RuntimeError(
                f"full-side refit parity FAILED for {r}@L{layer} ({source}): recomputed "
                f"{obs_skill[(r, 'full')]:.8f} vs committed {committed_skill:.8f} "
                f"(|Δ|={diff:.2e} > {REFIT_PARITY_TOL}) — the paired DV's full side is "
                "unstable (plan v15 kill criterion 3; failure_class: data)"
            )
        logger.info(
            "[refit_parity] %s@L%d (%s) |Δ|=%.2e OK; empty skill %.4f",
            r,
            layer,
            source,
            diff,
            obs_skill[(r, "empty")],
        )

    rng = np.random.default_rng(args.seed)
    B = args.n_boot
    idx = rng.integers(0, n, size=(B, n))  # ONE shared index matrix — paired everywhere
    d_draws = np.zeros((len(rows), B))
    per_row: dict[str, dict] = {}
    for ri, r in enumerate(rows):
        sk = {}
        for side in ("full", "empty"):
            res, tot = ss[(r, side)]
            rs = res[idx].sum(axis=1)  # (B,)
            ts = tot[idx].sum(axis=1)
            sk[side] = np.where(ts < 1e-12, np.nan, 1.0 - rs / ts)
        d_draws[ri] = sk["full"] - sk["empty"]
        d_obs = float(obs_skill[(r, "full")] - obs_skill[(r, "empty")])
        st = _stat_summary(d_obs, d_draws[ri])
        ci_lo, ci_hi = st["ci95"]
        per_row[r] = {
            **st,
            "committed_best_layer": committed_ref[r][0],
            "observed_full_skill": obs_skill[(r, "full")],
            "observed_empty_skill": obs_skill[(r, "empty")],
            "contained_within_margin": bool(
                ci_lo > -EQUIVALENCE_MARGIN and ci_hi < EQUIVALENCE_MARGIN
            ),
            "verdict": _verdict(ci_lo, ci_hi),
        }
        logger.info(
            "[paired] %s Δ(full−empty) obs %+0.4f CI95 [%+0.4f, %+0.4f] -> %s",
            r,
            d_obs,
            ci_lo,
            ci_hi,
            per_row[r]["verdict"],
        )
    # Familywise read (plan v15 §6 nulls item 2): center the shared-index joint
    # draws per row (preserves cross-row dependence), count draws where ANY
    # |Δ̄_centered| ≥ the margin.
    centered = d_draws - np.nanmean(d_draws, axis=1, keepdims=True)  # (n_rows, B)
    fam_hits = np.any(np.abs(centered) >= EQUIVALENCE_MARGIN, axis=0)
    familywise = {
        "margin": EQUIVALENCE_MARGIN,
        "n_rows": len(rows),
        "p_ge1_abs_centered_delta_ge_margin": float(np.nanmean(fam_hits)),
        "method": (
            "shared-index joint bootstrap draws centered per row (null of no effect, "
            "cross-row dependence preserved); P(any row |Δ̄_centered| ≥ margin)"
        ),
    }
    return {
        "pca_dim": pca_dim,
        "refit_parity": refit_parity,
        "per_row": per_row,
        "familywise": familywise,
        "draws_by_row": {r: [float(x) for x in d_draws[ri]] for ri, r in enumerate(rows)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Issue #810 header-echo mechanism read + paired full-vs-empty bootstrap"
    )
    ap.add_argument(
        "--he-summaries",
        default=None,
        help="he_summaries.pt (local path or HF path) — the EMPTY-answer side "
        "(REQUIRED outside --smoke)",
    )
    ap.add_argument(
        "--uh-summaries",
        default=UH_SUMMARIES_HF_FILE,
        help="round-3 uh_summaries.pt (local path or HF path) — the full side of the "
        "7 uh/bnd pairs",
    )
    ap.add_argument(
        "--committed-recon",
        default=str(
            PROJECT_ROOT / "eval_results" / "issue_810" / "reconstruction_skill_by_summary.json"
        ),
        help="round-1 committed recon JSON (im_end/turn_nl best layers + refit targets)",
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
        help="round-3 committed recon JSON (uh/bnd best layers + refit targets)",
    )
    ap.add_argument(
        "--rows",
        nargs="*",
        default=None,
        help="pair-row subset (default: all 9 HE rows; smoke default: uh_nl turn_nl — "
        "one round-3- and one round-1-targeted row)",
    )
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "eval_results" / "issue_810" / "header-echo-ablation-capture"),
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    rows = args.rows or (["uh_nl", "turn_nl"] if args.smoke else list(HE_SUMMARY_NAMES))
    _assert_row_coverage(rows)  # plan v15 Must-Fix: fires BEFORE any download

    man = load_json(hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset"))
    ctx_ids = context_ids_from_manifest(man)
    _free, capture_layers = _load_free_summaries()
    logger.info("[phase=load] rows=%s n_ctx=%d (smoke=%s)", rows, len(ctx_ids), args.smoke)

    full, committed_ref = _full_side_sources(rows, ctx_ids, capture_layers, args)
    empty, _he_meta = _load_empty_side(args, rows, ctx_ids, capture_layers)
    self_paired = empty is None
    if self_paired:
        empty = full  # smoke-only self-pair (Δ ≡ 0), loudly logged above

    cc = _load_cc(ctx_ids, capture_layers)

    logger.info("[phase=mechanism] centered cosine + pooled R² per (row × layer)")
    mech = _mechanism_read(full, empty, rows, ctx_ids, capture_layers, committed_ref)
    out_dir = Path(args.out)
    dump_json(
        {
            "dv": "cross_context_centered_similarity_full_vs_empty",
            "rows": rows,
            "n_contexts": len(ctx_ids),
            "capture_layers": capture_layers,
            "self_paired_smoke": self_paired,
            "by_row": mech,
            "reproducibility": reproducibility_metadata(),
            "smoke": args.smoke,
        },
        out_dir / "mechanism_cosine_r2.json",
    )
    logger.info("[phase=mechanism] wrote %s", out_dir / "mechanism_cosine_r2.json")

    logger.info("[phase=paired] full-side refit parity + shared-index bootstrap")
    paired = _paired_bootstrap(args, full, empty, rows, ctx_ids, capture_layers, committed_ref, cc)
    dump_json(
        {
            "dv": "paired_bootstrap_delta_skill_full_minus_empty",
            "method": (
                "per-context (ss_res, ss_tot) decompositions per (row, side) at each row's "
                "COMMITTED best layer, canonical manifest context order; ONE shared (B, n) "
                "context-resample index matrix paired across every row and BOTH sides; "
                "full side refit-parity-asserted vs the committed skills (≤1e-6); verdicts "
                "per plan v15 §3 (equivalence margin ±0.02)"
            ),
            "rows": rows,
            "n_contexts": len(ctx_ids),
            "n_boot": args.n_boot,
            "seed": args.seed,
            "self_paired_smoke": self_paired,
            **paired,
            "reproducibility": reproducibility_metadata(),
            "smoke": args.smoke,
        },
        out_dir / "paired_full_minus_empty.json",
    )
    logger.info("[phase=paired] wrote %s", out_dir / "paired_full_minus_empty.json")
    logger.info("[phase=done] he mechanism driver complete (%d rows)", len(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
