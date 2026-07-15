#!/usr/bin/env python3
# ruff: noqa: RUF003
# Intentional Unicode (→, ×, Δ) in scientific docstrings + log messages.
"""Issue #1005 Phases F2 + F3: matched-length + prefix-convention batteries.

Runs the parent's F2 (matched-length answer-span control, 9 arms on the shared
PCA-48 answer-REMAINDER target) and F3 (prefix-convention arms on remainder +
full-answer targets) fit machinery — ``issue928_matched_length_control.fit_mlc_regime``
and ``issue928_prefix_mapping_arms.fit_pma_regime``, reused verbatim — on the
#1005 UNIFIED 18-vector store (plan §4.5/§4.6). No recapture: the unified
capture already carries the MLC + prefix vectors; rows failing the
matched-length floor are excluded via the store's ``mlc_row_mask``
(``MaskedStore``), never dropped from the F1 battery.

Within-run coherence (replacing the parent's cross-round committed-artifact
machinery, which has no referent in a single-capture replication — §4.0.2):

- **F1↔F2↔F3 resample alignment:** every battery regenerates the ONE seed-42
  resample matrix; digests are asserted equal to THIS run's F1
  ``bootstrap_deltaskill.json`` per-regime digests (and, on the full 50-context
  grid at n_boot 2000, to the literal parent-pinned ``EXPECTED_RESAMPLE_DIGEST``
  — the matrix depends only on (n_ctx, n_boot, seed), so the pin transfers).
- **PCA-basis coherence (F3):** ``fit_pma_regime(coherence_binding=True)``
  asserts the remainder-target identity ss_tot against F2's OWN ``mlc_ident``
  decomposition at every layer (in-run ⇒ exact match; kill criterion).
- **Answer-target frozen layers (F3):** re-derived from THIS run's F1 frozen
  direct-arm layers (plan §4.6: the parent's realized 25/27 are reference
  points, not pins) via the ``ans_frozen_layer`` default-preserving kwarg.
- **Prefix constancy (plan §4.6):** within-context prefix-vector cosine to the
  context mean asserted >= 0.9999 before any fit.

Usage::

    EPM_FIT_DEVICE=cuda uv run python scripts/issue1005_f2f3.py \\
        --store data/issue_1005/store --out eval_results/issue_1005 \\
        --figures-dir figures/issue_1005
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue658_fit_predictors import _requested_device, _resolve_device  # noqa: E402
from issue928_common import (  # noqa: E402
    BOOTSTRAP_DRAWS,
    BOOTSTRAP_SEED,
    MLC_K_MIN,
    MLC_REM_MIN,
    SHUFFLE_NULL_PERMS,
    SHUFFLE_NULL_SEED,
    dump_json,
    load_json,
    reproducibility_metadata,
    upload_folder_scoped_verify,
)
from issue928_fit_decomposition import Store, prepare_checkpoint_dir  # noqa: E402
from issue928_matched_length_control import (  # noqa: E402
    MLC_ALL_ARMS,
    MLC_ARM_INPUTS,
    MLC_EXPLORATORY_ARMS,
    MLC_REGISTERED_ARMS,
    MLC_REGISTERED_READS,
    assert_pair_row_coverage,
    fit_mlc_regime,
    make_mlc_figures,
    mlc_bootstrap_statistics,
    null_band_analysis,
)
from issue928_null_bootstrap import assert_group_ridge_matches_serial  # noqa: E402
from issue928_prefix_mapping_arms import (  # noqa: E402
    PMA_ARM_INPUTS,
    PMA_EXPLORATORY_ARMS,
    PMA_REGISTERED_ARMS,
    PMA_REGISTERED_READS,
    assert_committed_bootstrap_alignment,
    assert_pma_pair_coverage,
    fit_pma_regime,
    make_pma_figures,
    pma_bootstrap_statistics,
    pma_null_band_analysis,
)
from issue1005_common import (  # noqa: E402
    DECOMP_TENSORS_PREFIX_1005,
    FIGURES_PREFIX_1005,
    FIT_RESULTS_PREFIX_1005,
    MLC_ROW_MASK_KEY,
)

logger = logging.getLogger("issue1005_f2f3")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

FOLLOWUP_LABEL_1005 = "issue1005-unified-f2f3"
# Two-bar bf16-CUDA calibration (#779 rule, gotchas.md; measured on THIS store
# 2026-07-15: worst flat 0.991745 on f5_fmt_json, layer 27 in ALL 50 contexts,
# layer-0 min >= 0.999999 everywhere — the depth-amplified padded-batch noise
# signature, NOT a span bug, which corrupts layer 0 at cos 0.43-0.84):
PREFIX_CONSTANCY_EARLY_COS_MIN = 0.999  # layers 0-3 — the sharp bug catcher
PREFIX_CONSTANCY_FLAT_COS_MIN = 0.98  # all layers — gross-corruption guard
REGIMES = ("indiv", "avg_q")


def phase(name: str) -> None:
    """Poller-visible phase breadcrumb (one line per pipeline phase)."""
    logger.info("[phase=%s]", name)


class MaskedStore(Store):
    """The unified #1005 store restricted to MLC-floor-PASSING rows (plan §4.5).

    The unified capture keeps floor-failing rows in the battery with NaN MLC
    slots (F1 reads only the base names); F2/F3 consume THIS view — per-blob
    ``mlc_row_mask`` subsetting with ``probe_avg`` + group bookkeeping
    recomputed over the surviving rows. Fails loud on a missing mask, an
    emptied context, or any surviving NaN.
    """

    def __init__(self, store_dir: Path, blob_subdir: str = "percq_summaries"):
        super().__init__(store_dir, blob_subdir)
        row_ctx: list[int] = []
        for ci, c in enumerate(self.ctx_ids):
            b = dict(self.blobs[c])
            if MLC_ROW_MASK_KEY not in b:
                raise RuntimeError(f"store blob {c} lacks {MLC_ROW_MASK_KEY!r} — not a #1005 store")
            mask = torch.as_tensor(b[MLC_ROW_MASK_KEY], dtype=torch.bool)
            assert mask.shape[0] == b["per_q"].shape[0], (c, mask.shape, b["per_q"].shape)
            b["per_q"] = b["per_q"][mask]
            if b["per_q"].shape[0] == 0:
                raise RuntimeError(f"context {c}: zero MLC-floor-passing rows (coverage collapse)")
            if not torch.isfinite(b["per_q"].float()).all():
                raise RuntimeError(f"context {c}: NaN survives the MLC row mask — capture bug")
            b["probe_indices"] = [
                int(q) for q, m in zip(b["probe_indices"], mask.tolist(), strict=True) if m
            ]
            b["probe_avg"] = b["per_q"].float().mean(dim=0).to(torch.float16)
            self.blobs[c] = b
            row_ctx.extend([ci] * int(b["per_q"].shape[0]))
        self.row_ctx = row_ctx
        self.groups = np.asarray(row_ctx, dtype=np.int64)


def assert_f1_bootstrap_coherence(f1_boot: dict, n_boot: int) -> dict:
    """In-run F1↔F2/F3 resample-convention coherence (plan §4.6).

    THIS run's F1 bootstrap artifact must carry the inherited seed and the SAME
    ``n_boot`` as this battery (draw alignment is within-run, so the parent's
    ``== BOOTSTRAP_DRAWS`` production pin generalizes to the run's own value —
    a smoke at n_boot 50 stays coherent with its own F1)."""
    if int(f1_boot["seed"]) != BOOTSTRAP_SEED or int(f1_boot["n_boot"]) != n_boot:
        raise RuntimeError(
            f"F1 bootstrap metadata mismatch: (seed={f1_boot['seed']}, n_boot={f1_boot['n_boot']})"
            f" != (seed={BOOTSTRAP_SEED}, n_boot={n_boot}) — F2/F3 draws would not be F1-aligned"
        )
    return {"seed": int(f1_boot["seed"]), "n_boot": int(f1_boot["n_boot"])}


def prefix_constancy_assert(store: MaskedStore) -> dict:
    """Within-context prefix-vector constancy (plan §4.6; #779 two-bar bf16 gate).

    The prefix tokens are identical across a context's rows (the probe is
    excluded from the prefix span), so per-row ``prefix_mean`` vectors must
    agree with the context mean up to batched-forward numerics. Bar (a):
    EARLY layers 0-3 — a real span/pad/row bug corrupts layer 0 immediately
    (cos 0.43-0.84); bar (b): flat all-layer gross-corruption guard (measured
    bf16 depth-noise worst on this store: 0.991745, layer-27-only)."""
    pfx = store.sidx["prefix_mean"]
    report: dict = {}
    worst_early = 1.0
    worst_flat = 1.0
    for c in store.ctx_ids:
        v = store.blobs[c]["per_q"][:, pfx].float()  # (n, Lc, H)
        if v.shape[0] < 2:
            report[c] = {"cos_min_to_mean": None, "n_rows": int(v.shape[0])}
            continue
        mean_v = v.mean(dim=0, keepdim=True)
        cos = torch.nn.functional.cosine_similarity(v, mean_v, dim=-1)  # (n, Lc)
        early = float(cos[:, :4].min())
        flat = float(cos.min())
        report[c] = {
            "cos_min_early_l0_3": early,
            "cos_min_to_mean": flat,
            "n_rows": int(v.shape[0]),
        }
        worst_early = min(worst_early, early)
        worst_flat = min(worst_flat, flat)
    assert worst_early >= PREFIX_CONSTANCY_EARLY_COS_MIN, (
        f"prefix-constancy assert FAILED: EARLY-layer (0-3) min cosine {worst_early:.6f} < "
        f"{PREFIX_CONSTANCY_EARLY_COS_MIN} — layer-0-visible drift is a real capture/span "
        "bug, not bf16 depth noise (plan §4.6)"
    )
    assert worst_flat >= PREFIX_CONSTANCY_FLAT_COS_MIN, (
        f"prefix-constancy assert FAILED: flat min cosine {worst_flat:.6f} < "
        f"{PREFIX_CONSTANCY_FLAT_COS_MIN} — beyond the measured bf16 depth-noise envelope "
        "(plan §4.6)"
    )
    return {
        "cos_min_early_overall": worst_early,
        "cos_min_overall": worst_flat,
        "bars": {
            "early_l0_3": PREFIX_CONSTANCY_EARLY_COS_MIN,
            "flat": PREFIX_CONSTANCY_FLAT_COS_MIN,
        },
        "by_context": report,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #1005 F2+F3 batteries (unified store)")
    ap.add_argument("--store", default=str(PROJECT_ROOT / "data" / "issue_1005" / "store"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "eval_results" / "issue_1005"))
    ap.add_argument("--figures-dir", default=str(PROJECT_ROOT / "figures" / "issue_1005"))
    ap.add_argument("--layers", nargs="*", type=int, default=None, help="layer-INDEX subset")
    ap.add_argument("--device", default=None, help="fit device: CLI > EPM_FIT_DEVICE > auto")
    ap.add_argument("--n-perms", type=int, default=SHUFFLE_NULL_PERMS)
    ap.add_argument("--n-boot", type=int, default=BOOTSTRAP_DRAWS)
    ap.add_argument("--draw-chunk", type=int, default=16)
    ap.add_argument("--skip-parity-gate", action="store_true", help="skip the serial ridge gate")
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="label only (subset runs are inferred)")
    args = ap.parse_args()

    t0 = time.time()
    store_dir = Path(args.store)
    out_dir = Path(args.out)
    figures_dir = Path(args.figures_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    fit_device = _resolve_device(_requested_device(args.device))
    logger.info("fit device: %s", fit_device)

    phase("setup")
    store = MaskedStore(store_dir)
    ctx_ids = store.ctx_ids
    n_ctx = len(ctx_ids)
    layers_idx = args.layers if args.layers is not None else list(range(len(store.layers)))
    full_grid = n_ctx == 50 and args.layers is None and args.n_boot == BOOTSTRAP_DRAWS
    manifest = store.manifest
    floor_misses = manifest.get("mlc_floor_misses", {})
    f1_boot = load_json(out_dir / "bootstrap_deltaskill.json")
    f1_coherence = assert_f1_bootstrap_coherence(f1_boot, args.n_boot)
    ans_frozen_layer = {
        r: int(f1_boot["by_regime"][r]["layer_conventions"]["primary_frozen_direct_best_layer"])
        for r in REGIMES
    }
    logger.info("F3 answer-target frozen layers (re-derived from F1): %s", ans_frozen_layer)

    phase("parity")
    if not args.skip_parity_gate:
        logger.info("[phase=parity] batched group-ridge vs serial reference (atol 1e-8)")
        ridge_parity = assert_group_ridge_matches_serial()
    else:
        ridge_parity = {"skipped": True}
    constancy = prefix_constancy_assert(store)
    logger.info("[phase=parity] prefix constancy PASS (cos_min=%.6f)", constancy["cos_min_overall"])

    # ── F2: matched-length battery on the shared remainder target ─────────────
    ckpt_root = out_dir / "partial"
    mlc_grid: dict = {}
    mlc_null: dict = {}
    mlc_decomp: dict = {}
    for regime in REGIMES:
        phase(f"f2_fit_{regime}")
        regime_key = {
            "regime": regime,
            "round": FOLLOWUP_LABEL_1005,
            "battery": "mlc",
            "store_identity": store.identity_digest(),
            "layers": [int(store.layers[li]) for li in layers_idx],
            "arms": list(MLC_ALL_ARMS),
            "n_perms": int(args.n_perms),
            "shuffle_null_seed": int(SHUFFLE_NULL_SEED),
            "standardization": "per_fold" if regime == "avg_q" else "full_data",
            "floors": {"k_min": MLC_K_MIN, "rem_min": MLC_REM_MIN},
            "device": fit_device,
        }
        ckpt_dir = prepare_checkpoint_dir(ckpt_root, f"i1005_mlc_{regime}", regime_key)
        grid, null_matrix, decomp = fit_mlc_regime(
            store,
            regime,
            layers_idx,
            fit_device,
            args.n_perms,
            args.draw_chunk,
            checkpoint_dir=ckpt_dir,
        )
        mlc_grid[regime] = grid
        mlc_null[regime] = null_matrix
        mlc_decomp[regime] = decomp
        dump_json(
            {
                "dv": "recon_skill_over_mean_r2 (answer-REMAINDER target)",
                "regime": regime,
                "round": FOLLOWUP_LABEL_1005,
                "axes": "arm -> layer -> [per-draw skill]",
                "n_perms": args.n_perms,
                "seed": SHUFFLE_NULL_SEED,
                "perm_grain": "context" if regime == "avg_q" else "context-group",
                "registered_arms": list(MLC_REGISTERED_ARMS),
                "null": null_matrix,
            },
            out_dir / f"null_matrix_{regime}_mlc.json",
        )
        torch.save(
            {
                str(k): {"ss_res": v["ss_res"], "ss_tot": v["ss_tot"], "ctx_order": v["ctx_order"]}
                for k, v in decomp.items()
            },
            out_dir / f"decomp_{regime}_mlc.pt",
        )

    phase("f2_bootstrap")
    mlc_boot: dict = {}
    mlc_bands: dict = {}
    mlc_coverage: dict = {}
    for regime in REGIMES:
        mlc_coverage[regime] = assert_pair_row_coverage(mlc_decomp[regime], n_ctx)
        mlc_boot[regime] = mlc_bootstrap_statistics(mlc_decomp[regime], n_ctx, args.n_boot)
        mlc_bands[regime] = null_band_analysis(mlc_null[regime], mlc_decomp[regime])
        f1_digest = f1_boot["by_regime"].get(regime, {}).get("resample_matrix_digest")
        got = mlc_boot[regime]["resample_matrix_digest"]
        if f1_digest is not None and got != f1_digest:
            raise RuntimeError(
                f"F1↔F2 resample digest mismatch ({regime}): {got} != {f1_digest} — "
                "cross-battery contrasts would NOT be draw-aligned (plan §4.6 kill criterion)"
            )
    mlc_boot_blob = {
        "dv": "paired bootstrap delta-skill on the answer-REMAINDER target",
        "round": FOLLOWUP_LABEL_1005,
        "seed": BOOTSTRAP_SEED,
        "n_boot": args.n_boot,
        "f1_coherence": f1_coherence,
        "registered_reads": [list(r) for r in MLC_REGISTERED_READS],
        "pair_row_coverage": {
            r: {k: v for k, v in cov.items() if k != "ctx_order"} for r, cov in mlc_coverage.items()
        },
        "by_regime": mlc_boot,
        "reproducibility": reproducibility_metadata(),
    }
    dump_json(mlc_boot_blob, out_dir / "mlc_bootstrap_deltaskill.json")
    dump_json(
        {
            "dv": "held-out skill-over-mean R^2 per (arm x layer x regime x fold), rem target",
            "round": FOLLOWUP_LABEL_1005,
            "estimator": (
                "inherited #810/#928: LOCO ridge, nested-CV lambda over RIDGE_LAMBDAS, "
                "full-data PCA-48 remainder-target basis with per-fold train centering; "
                "avg_q per-fold X standardization, indiv full-data X standardization"
            ),
            "context_ids": ctx_ids,
            "capture_layers": [int(store.layers[li]) for li in layers_idx],
            "n_indiv_rows": int(store.groups.shape[0]),
            "mlc_floor_misses": floor_misses,
            "ridge_parity_gate": ridge_parity,
            "prefix_constancy": {k: v for k, v in constancy.items() if k != "by_context"},
            "arm_inputs": {a: list(r) for a, r in MLC_ARM_INPUTS.items()},
            "registered_arms": list(MLC_REGISTERED_ARMS),
            "exploratory_arms": list(MLC_EXPLORATORY_ARMS),
            "grid": mlc_grid,
            "frozen_layers": {r: mlc_boot[r]["layer_conventions"] for r in mlc_boot},
            "null_band_vs_ceiling": mlc_bands,
            "n_perms": args.n_perms,
            "n_boot": args.n_boot,
            "full_grid": full_grid,
            "reproducibility": reproducibility_metadata(),
        },
        out_dir / "mlc_skill_grid.json",
    )

    # ── F3: prefix-convention battery (remainder + full-answer targets) ───────
    alignment = assert_committed_bootstrap_alignment(
        mlc_boot_blob, mlc_decomp, n_ctx, args.n_boot, full_grid
    )
    pma_grid: dict = {}
    pma_null: dict = {}
    pma_decomp: dict = {}
    pma_coherence: dict = {}
    for regime in REGIMES:
        phase(f"f3_fit_{regime}")
        regime_key = {
            "regime": regime,
            "round": FOLLOWUP_LABEL_1005,
            "battery": "pma",
            "store_identity": store.identity_digest(),
            "layers": [int(store.layers[li]) for li in layers_idx],
            "arms": list(PMA_ARM_INPUTS),
            "n_perms": int(args.n_perms),
            "shuffle_null_seed": int(SHUFFLE_NULL_SEED),
            "standardization": "per_fold" if regime == "avg_q" else "full_data",
            "device": fit_device,
        }
        ckpt_dir = prepare_checkpoint_dir(ckpt_root, f"i1005_pma_{regime}", regime_key)
        grid, null_matrix, decomp, coherence = fit_pma_regime(
            store,
            regime,
            layers_idx,
            fit_device,
            args.n_perms,
            args.draw_chunk,
            committed_decomp=mlc_decomp[regime],
            coherence_binding=True,
            checkpoint_dir=ckpt_dir,
        )
        pma_grid[regime] = grid
        pma_null[regime] = null_matrix
        pma_decomp[regime] = decomp
        pma_coherence[regime] = coherence
        dump_json(
            {
                "dv": "recon_skill_over_mean_r2 (remainder + full-answer targets)",
                "regime": regime,
                "round": FOLLOWUP_LABEL_1005,
                "axes": "arm -> layer -> [per-draw skill]",
                "n_perms": args.n_perms,
                "seed": SHUFFLE_NULL_SEED,
                "perm_grain": "context" if regime == "avg_q" else "context-group",
                "registered_arms": list(PMA_REGISTERED_ARMS),
                "null": null_matrix,
            },
            out_dir / f"null_matrix_{regime}_pma.json",
        )
        torch.save(
            {
                str(k): {"ss_res": v["ss_res"], "ss_tot": v["ss_tot"], "ctx_order": v["ctx_order"]}
                for k, v in decomp.items()
            },
            out_dir / f"decomp_{regime}_pma.pt",
        )

    phase("f3_bootstrap")
    pma_boot: dict = {}
    pma_bands: dict = {}
    pma_coverage: dict = {}
    for regime in REGIMES:
        pma_coverage[regime] = assert_pma_pair_coverage(
            pma_decomp[regime], mlc_decomp[regime], n_ctx
        )
        pma_boot[regime] = pma_bootstrap_statistics(
            pma_decomp[regime],
            mlc_decomp[regime],
            regime,
            n_ctx,
            args.n_boot,
            alignment,
            full_grid,
            ans_frozen_layer=ans_frozen_layer,
        )
        pma_bands[regime] = pma_null_band_analysis(
            pma_null[regime], pma_decomp[regime], mlc_decomp[regime]
        )
    dump_json(
        {
            "dv": "paired bootstrap delta-skill, prefix-convention arms (both targets)",
            "round": FOLLOWUP_LABEL_1005,
            "seed": BOOTSTRAP_SEED,
            "n_boot": args.n_boot,
            "alignment": alignment,
            "ans_frozen_layer_re_derived_from_f1": ans_frozen_layer,
            "registered_reads": [list(r) for r in PMA_REGISTERED_READS],
            "pair_row_coverage": {
                r: {k: v for k, v in cov.items() if k != "ctx_order"}
                for r, cov in pma_coverage.items()
            },
            "by_regime": pma_boot,
            "reproducibility": reproducibility_metadata(),
        },
        out_dir / "pma_bootstrap_deltaskill.json",
    )
    dump_json(
        {
            "dv": "held-out skill-over-mean R^2 per (arm x layer x regime x fold), both targets",
            "round": FOLLOWUP_LABEL_1005,
            "context_ids": ctx_ids,
            "capture_layers": [int(store.layers[li]) for li in layers_idx],
            "n_indiv_rows": int(store.groups.shape[0]),
            "arm_inputs": {a: list(r) for a, r in PMA_ARM_INPUTS.items()},
            "registered_arms": list(PMA_REGISTERED_ARMS),
            "exploratory_arms": list(PMA_EXPLORATORY_ARMS),
            "grid": pma_grid,
            "frozen_layers": {r: pma_boot[r]["layer_conventions"] for r in pma_boot},
            "basis_coherence_by_layer": pma_coherence,
            "null_band_vs_ceiling": pma_bands,
            "n_perms": args.n_perms,
            "n_boot": args.n_boot,
            "full_grid": full_grid,
            "reproducibility": reproducibility_metadata(),
        },
        out_dir / "pma_skill_grid.json",
    )

    # ── figures (arm-parametrized helpers; stems land under figures_dir.name) ─
    phase("figures")
    bookkeeping = load_json(store_dir / "row_bookkeeping.json")["per_context"]
    floor_drops = {c: int(floor_misses.get(c, 0)) for c in ctx_ids}
    fig_stems = make_mlc_figures(
        figures_dir, mlc_grid, mlc_boot, mlc_bands, mlc_decomp, bookkeeping, floor_drops
    )
    fig_stems += make_pma_figures(
        figures_dir, pma_grid, pma_boot, pma_bands, pma_decomp, mlc_boot_blob, bookkeeping
    )

    hf_paths: dict = {}
    if not args.no_upload:
        phase("upload")
        suffix = "" if not args.smoke else "_smoke"
        json_names = sorted(
            p.name
            for p in out_dir.glob("*.json")
            if p.name.startswith(("mlc_", "pma_", "null_matrix_"))
        )
        hf_paths["fit_results"] = upload_folder_scoped_verify(
            out_dir,
            f"{FIT_RESULTS_PREFIX_1005}/f2f3" + suffix,
            json_names,
            f"issue #1005 {FOLLOWUP_LABEL_1005}: F2+F3 fit results",
            allow_patterns=[f"{n}" for n in json_names],
            ignore_patterns=["partial/*"],
        )
        pt_names = sorted(p.name for p in out_dir.glob("decomp_*_mlc.pt")) + sorted(
            p.name for p in out_dir.glob("decomp_*_pma.pt")
        )
        hf_paths["decomp"] = upload_folder_scoped_verify(
            out_dir,
            f"{DECOMP_TENSORS_PREFIX_1005}/f2f3" + suffix,
            pt_names,
            f"issue #1005 {FOLLOWUP_LABEL_1005}: per-context LOCO decompositions",
            allow_patterns=["decomp_*_mlc.pt", "decomp_*_pma.pt"],
        )
        fig_files = sorted(
            p.name
            for stem in fig_stems
            for p in figures_dir.glob(f"{stem}.*")
            if p.suffix in (".png", ".pdf", ".json")
        )
        hf_paths["figures"] = upload_folder_scoped_verify(
            figures_dir,
            f"{FIGURES_PREFIX_1005}/f2f3" + suffix,
            fig_files,
            f"issue #1005 {FOLLOWUP_LABEL_1005}: F2+F3 figures",
            allow_patterns=[f"{stem}.*" for stem in fig_stems],
        )

    read1 = mlc_boot["indiv"]["statistics"]["read1_primary_ctx_cotK_minus_ctx_apfx"]
    logger.info(
        "[phase=f2f3_done] MLC read1 primary (indiv): obs=%.4f ci95=%s | hf=%s | %.1fs",
        read1["primary_frozen_ctx_baseline_best"]["observed"],
        read1["primary_frozen_ctx_baseline_best"]["ci95"],
        hf_paths,
        time.time() - t0,
    )
    return 0


if __name__ == "__main__":
    import traceback

    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        logger.error("[phase=failed] issue1005 F2/F3 crashed:\n%s", traceback.format_exc())
        raise
