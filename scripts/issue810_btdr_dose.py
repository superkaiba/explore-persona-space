#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, ², ≥, ×, −) in scientific docstrings + log messages.
"""Issue #810 `_btdr` paired dose-response driver (plan v18 §4.6 item 4).

The boundary-truncation dose round's CPU-chain analysis — a THIN driver over
``issue810_he_mechanism.py``'s batched round-4 primitives (imported, never
reimplemented):

(a) **Paired dose bootstrap (H1/H2-btdr PRIMARY).** Δskill(full − truncated)
    per (row × k) at the full side's COMMITTED best layer, via the round-4
    ``_paired_bootstrap`` — per k, ONE shared (B, n) context-resample index
    matrix drawn from ``default_rng(seed)`` with the SAME seed 42, so the
    IDENTICAL index matrix is shared across every (row × k × side) cell by
    construction (within-context-coherent dose traces). Full-side refit parity
    ≤ 1e-6 vs the committed round-1/round-3 skills fires inside every call.
(b) **Empty-side (k=0) refit parity.** The he pack's LOCO skill at each row's
    committed layer must match the round-4 committed
    ``reconstruction_skill_header_echo.json`` value ≤ 1e-6 (plan v18 kill
    criterion: the paired DV's k=0 comparator must be the committed numbers).
(c) **Familywise reads.** The round-4 centered-joint-draw construction over
    the 27-cell family (9 rows × 3 k) AND the 6-cell primary sub-family
    (turn_nl + uh_im_start × 3 k); draws persist in the output JSON.
(d) **Mechanism companion (H3-btdr).** ``_mechanism_read`` per k — centered
    cosine full-vs-truncated per (row × k × layer) + per-context values at
    committed layers.
(e) **Descriptive extras for the figures:** absolute truncated-side skill
    draws (hero CI whiskers) + per-context Δ (spaghetti) per (row × k incl.
    k=0), all from the SAME per-context (ss_res, ss_tot) decompositions;
    the round-4 committed k=0 paired deltas are ECHOED (never re-derived).

ROW-COVERAGE assert at driver start (plan v18 §4.6 item 4, the round-4
Must-Fix): ``set(rows) ⊆ union(full-side sources)``.

Usage::

    # production (cpu-mid, after the GPU phase lands the 3 btdr packs on HF):
    uv run python scripts/issue810_btdr_dose.py \\
        --out eval_results/issue_810/boundary-truncation-dose-response

    # smoke (2 rows, 200 draws; real full + k=0 sides from HF; a missing /
    # shape-mismatched btdr pack SELF-PAIRS that k's side (Δ ≡ 0), loudly):
    uv run python scripts/issue810_btdr_dose.py --smoke --n-boot 200 \\
        --out /tmp/i810_btdr_smoke/dose
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
from issue810_bootstrap_deltaskill import _per_context_decomposition, _skill  # noqa: E402
from issue810_common import (  # noqa: E402
    BTDR_HF_RESULTS_PREFIX,
    BTDR_SUMMARIES_HF_FILE_TMPL,
    BTDR_TRUNCATE_FRACS,
    HE_SUMMARIES_HF_FILE,
    HE_SUMMARY_NAMES,
    HF_DATA_REPO,
    I658_STORE_MANIFEST,
    PCA_TARGET_DIM_CAP,
    UH_SUMMARIES_HF_FILE,
    btdr_pct,
    context_ids_from_manifest,
    dump_json,
    load_json,
    reproducibility_metadata,
    validate_uh_pack,
)
from issue810_fit_readout import _load_uh_summaries  # noqa: E402
from issue810_fit_reconstruction import _load_cc, _load_free_summaries  # noqa: E402
from issue810_he_mechanism import (  # noqa: E402
    EQUIVALENCE_MARGIN,
    REFIT_PARITY_TOL,
    _assert_row_coverage,
    _full_side_sources,
    _load_empty_side,
    _mechanism_read,
    _paired_bootstrap,
)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue810_btdr_dose")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# The two round-4 positive singles — the pre-registered primary sub-family
# (plan v18 §3: Δ(k=25) containment + Δ(k=50) CI reads bind on these).
PRIMARY_SINGLES: tuple[str, ...] = ("turn_nl", "uh_im_start")


def _load_btdr_side(
    spec: str, k: float, rows: list[str], ctx_ids: list[str], capture_layers: list[int], smoke: bool
):
    """The truncated side for one k from ``btdr_summaries_k{pct}.pt``.

    Production: the pack must carry ``truncate_frac == k`` provenance (a wrong-k
    or ablated/uh pack refuses) and passes ``validate_uh_pack``. Smoke fallback
    (--smoke only, the ``_load_empty_side`` precedent): a missing /
    smoke-provenance / shape- or k-mismatched pack SELF-PAIRS this k's side to
    the full side (Δ ≡ 0 — machinery, asserts, and JSON contracts still
    exercised end-to-end on real full-side data), LOUDLY logged, never silent.
    Returns ``(side_rows | None, meta)``; None = self-pair.
    """
    try:
        rows_d, cov, meta = _load_uh_summaries(spec)
    except Exception:
        if not smoke:
            raise
        logger.warning(
            "[smoke] btdr pack for k=%s at %s unavailable — SELF-PAIRING (Δ ≡ 0)", k, spec
        )
        return None, {"self_paired": True}
    pack_k = meta.get("truncate_frac")
    if not smoke:
        if meta.get("ablate_answer"):
            raise RuntimeError(f"btdr pack for k={k} at {spec} is an ABLATED pack — wrong side")
        if pack_k is None or abs(float(pack_k) - float(k)) > 1e-9:
            raise RuntimeError(
                f"btdr pack truncate_frac mismatch for k={k}: pack carries {pack_k!r} "
                f"({spec}) — refusing to pair a wrong-dose capture"
            )
        validate_uh_pack(
            {r: {c: torch.from_numpy(v) for c, v in per.items()} for r, per in rows_d.items()},
            cov,
            meta,
            requested_rows=rows,
            ctx_ids=ctx_ids,
            expected_capture_layers=capture_layers,
        )
        return {r: rows_d[r] for r in rows}, meta
    usable = (
        pack_k is not None
        and abs(float(pack_k) - float(k)) <= 1e-9
        and all(
            r in rows_d
            and all(c in rows_d[r] for c in ctx_ids)
            and rows_d[r][ctx_ids[0]].shape[0] == len(capture_layers)
            for r in rows
        )
    )
    if usable:
        logger.info("[smoke] pairing k=%s against the provided btdr pack (%s)", k, spec)
        return {r: rows_d[r] for r in rows}, meta
    logger.warning(
        "[smoke] btdr pack for k=%s at %s is smoke-provenance/shape/k-mismatched "
        "(model=%s, smoke=%s, truncate_frac=%s) — SELF-PAIRING this side (Δ ≡ 0); the "
        "loader/schema path was still exercised on the real pack file",
        k,
        spec,
        meta.get("model"),
        meta.get("smoke"),
        pack_k,
    )
    return None, meta


def _side_decomps(
    sides: dict[str, dict],
    rows: list[str],
    ctx_ids: list[str],
    capture_layers: list[int],
    committed_ref: dict,
    cc: dict,
    pca_dim: int,
) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
    """(row, side) -> per-context (ss_res, ss_tot) at each row's committed layer.

    One closed-form LOCO decomposition per (row × side) via the imported
    round-4 ``_per_context_decomposition`` (batched internals, no per-draw
    loop) — the shared basis for the empty-side refit parity, the absolute
    skill draws, and the per-context Δ traces.
    """
    out: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for r in rows:
        li = capture_layers.index(committed_ref[r][0])
        Xc = np.stack([cc[c][li] for c in ctx_ids])
        for side_name, src in sides.items():
            Yv = np.stack([src[r][c][li] for c in ctx_ids])
            out[(r, side_name)] = _per_context_decomposition(Xc, Yv, pca_dim)
    return out


def _empty_side_refit_parity(
    decomps: dict, rows: list[str], committed_ref: dict, committed_recon_he: str
) -> dict[str, dict]:
    """(b) k=0 side refit parity vs round-4 committed skills ≤ 1e-6 (fail loud)."""
    he_points = load_json(committed_recon_he)["by_summary"]
    out: dict[str, dict] = {}
    for r in rows:
        layer = committed_ref[r][0]
        cell = next(c for c in he_points[r] if int(c["layer"]) == layer)
        committed = float(cell["ridge_skill"])
        recomputed = _skill(*decomps[(r, "k0")])
        diff = abs(recomputed - committed)
        out[r] = {
            "committed_best_layer": layer,
            "committed_skill": committed,
            "recomputed_skill": recomputed,
            "abs_diff": diff,
        }
        if diff > REFIT_PARITY_TOL:
            raise RuntimeError(
                f"EMPTY-side (k=0) refit parity FAILED for {r}@L{layer}: recomputed "
                f"{recomputed:.8f} vs round-4 committed {committed:.8f} "
                f"(|Δ|={diff:.2e} > {REFIT_PARITY_TOL}) — the dose trace's k=0 comparator "
                "is unstable (plan v18 kill criterion; failure_class: data)"
            )
        logger.info("[refit_parity_k0] %s@L%d |Δ|=%.2e OK", r, layer, diff)
    return out


def _abs_skill_and_percontext(
    decomps: dict, rows: list[str], side_names: list[str], idx: np.ndarray
) -> tuple[dict, dict]:
    """(e) absolute per-side skill {observed, ci95} + per-context skill/Δ traces.

    Draws use the SAME shared index matrix ``idx`` as the paired bootstrap
    (regenerated from the identical seed — deterministic given (seed, B, n)).
    """
    abs_skill: dict[str, dict[str, dict]] = {s: {} for s in side_names}
    per_ctx_skill: dict[str, dict[str, list[float]]] = {s: {} for s in side_names}
    for s in side_names:
        for r in rows:
            res, tot = decomps[(r, s)]
            rs = res[idx].sum(axis=1)
            ts = tot[idx].sum(axis=1)
            draws = np.where(ts < 1e-12, np.nan, 1.0 - rs / ts)
            abs_skill[s][r] = {
                "observed": _skill(res, tot),
                "ci95": [float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))],
            }
            pc = np.where(tot < 1e-12, np.nan, 1.0 - res / tot)
            per_ctx_skill[s][r] = [float(x) for x in pc]
    return abs_skill, per_ctx_skill


def _familywise(cells: dict[tuple[str, str], np.ndarray], label: str) -> dict:
    """Round-4 centered-joint-draw familywise read over a cell family.

    Center each cell's shared-index draws, count draws where ANY
    |Δ̄_centered| ≥ the ±0.02 margin (cross-cell dependence preserved by the
    shared index matrix).
    """
    mat = np.stack(list(cells.values()))  # (n_cells, B)
    centered = mat - np.nanmean(mat, axis=1, keepdims=True)
    hits = np.any(np.abs(centered) >= EQUIVALENCE_MARGIN, axis=0)
    return {
        "family": label,
        "margin": EQUIVALENCE_MARGIN,
        "n_cells": len(cells),
        "cells": [f"{r}@k{kk}" for (r, kk) in cells],
        "p_ge1_abs_centered_delta_ge_margin": float(np.nanmean(hits)),
        "method": (
            "shared-index joint bootstrap draws centered per (row × k) cell (null of no "
            "effect, cross-cell dependence preserved); P(any cell |Δ̄_centered| ≥ margin)"
        ),
    }


def _k0_committed_echo(committed_paired: str, rows: list[str], committed_ref: dict) -> dict:
    """ECHO the round-4 committed k=0 paired deltas (never re-derived — plan v18 §4.6).

    Asserts the committed layers match this run's paired-read slots (a drifted
    layer table would silently misalign the dose trace's k=0 point).
    """
    blob = load_json(committed_paired)
    per_row = blob.get("per_row") or {}
    out: dict[str, dict] = {}
    for r in rows:
        if r not in per_row:
            raise RuntimeError(
                f"committed round-4 paired JSON lacks row {r!r} ({committed_paired})"
            )
        rec = per_row[r]
        if int(rec["committed_best_layer"]) != int(committed_ref[r][0]):
            raise RuntimeError(
                f"k=0 committed layer mismatch for {r}: round-4 paired JSON has "
                f"L{rec['committed_best_layer']} vs this run's committed table "
                f"L{committed_ref[r][0]}"
            )
        out[r] = {
            k: rec.get(k)
            for k in (
                "observed",
                "ci95",
                "p_delta_le_0",
                "verdict",
                "committed_best_layer",
                "observed_full_skill",
                "observed_empty_skill",
                "contained_within_margin",
            )
        }
    out["familywise_round4"] = blob.get("familywise")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Issue #810 boundary-truncation dose-response: paired bootstrap + mechanism"
    )
    ap.add_argument(
        "--truncate-fracs",
        nargs="+",
        type=float,
        default=list(BTDR_TRUNCATE_FRACS),
        help="the interior k grid (default 0.25 0.5 0.75 — the registered production grid)",
    )
    ap.add_argument(
        "--btdr-summaries",
        nargs="*",
        default=None,
        help="per-k btdr_summaries_k{pct}.pt specs (local paths or HF data-repo paths), "
        "one per --truncate-fracs value in order; default = the HF production paths "
        f"{BTDR_HF_RESULTS_PREFIX}/{BTDR_SUMMARIES_HF_FILE_TMPL}",
    )
    ap.add_argument(
        "--he-summaries",
        default=HE_SUMMARIES_HF_FILE,
        help="he_summaries.pt (local path or HF path) — the k=0 (empty-answer) side",
    )
    ap.add_argument(
        "--uh-summaries",
        default=UH_SUMMARIES_HF_FILE,
        help="round-3 uh_summaries.pt (local path or HF path) — the full side of the 7 uh/bnd rows",
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
        "--committed-recon-he",
        default=str(
            PROJECT_ROOT
            / "eval_results"
            / "issue_810"
            / "header-echo-ablation-capture"
            / "reconstruction_skill_header_echo.json"
        ),
        help="round-4 committed recon JSON (the k=0 side's refit-parity targets)",
    )
    ap.add_argument(
        "--committed-paired",
        default=str(
            PROJECT_ROOT
            / "eval_results"
            / "issue_810"
            / "header-echo-ablation-capture"
            / "paired_full_minus_empty.json"
        ),
        help="round-4 committed paired JSON (the k=0 deltas ECHOED into the dose trace)",
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
        default=str(
            PROJECT_ROOT / "eval_results" / "issue_810" / "boundary-truncation-dose-response"
        ),
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download

    rows = args.rows or (["uh_nl", "turn_nl"] if args.smoke else list(HE_SUMMARY_NAMES))
    _assert_row_coverage(rows)  # plan v18 §4.6 item 4: fires BEFORE any download
    ks = [float(k) for k in args.truncate_fracs]
    if args.btdr_summaries:
        if len(args.btdr_summaries) != len(ks):
            raise SystemExit(
                f"--btdr-summaries needs one spec per --truncate-fracs value "
                f"({len(args.btdr_summaries)} vs {len(ks)})"
            )
        pack_specs = dict(zip(ks, args.btdr_summaries, strict=True))
    else:
        pack_specs = {
            k: f"{BTDR_HF_RESULTS_PREFIX}/{BTDR_SUMMARIES_HF_FILE_TMPL.format(pct=btdr_pct(k))}"
            for k in ks
        }
    # The registered production family (plan v18 §3): 9 rows × 3 k = 27 cells
    # + the 6-cell primary sub-family. Subsets are smoke-only.
    if not args.smoke and (
        set(rows) != set(HE_SUMMARY_NAMES) or sorted(ks) != sorted(BTDR_TRUNCATE_FRACS)
    ):
        raise SystemExit(
            "production runs use the registered 9-row × {0.25,0.5,0.75} family "
            "(27 cells) — pass --smoke for subsets"
        )

    man = load_json(hf_hub_download(HF_DATA_REPO, I658_STORE_MANIFEST, repo_type="dataset"))
    ctx_ids = context_ids_from_manifest(man)
    _free, capture_layers = _load_free_summaries()
    n = len(ctx_ids)
    pca_dim = min(PCA_TARGET_DIM_CAP, n - 2)
    logger.info(
        "[phase=load] rows=%s ks=%s n_ctx=%d (smoke=%s)", rows, ks, len(ctx_ids), args.smoke
    )

    full, committed_ref = _full_side_sources(rows, ctx_ids, capture_layers, args)
    empty, _he_meta = _load_empty_side(args, rows, ctx_ids, capture_layers)
    self_paired_k0 = empty is None
    if self_paired_k0:
        empty = full  # smoke-only self-pair (Δ ≡ 0), loudly logged by the loader
    sides: dict[str, dict] = {"full": full, "k0": empty}
    self_paired_ks: list[float] = []
    for k in ks:
        side, _meta = _load_btdr_side(pack_specs[k], k, rows, ctx_ids, capture_layers, args.smoke)
        if side is None:
            self_paired_ks.append(k)
            side = full
        sides[f"k{btdr_pct(k)}"] = side
    if not args.smoke and (self_paired_k0 or self_paired_ks):
        raise RuntimeError("self-paired sides are smoke-only (production packs are required)")

    cc = _load_cc(ctx_ids, capture_layers)

    # (b) + (e): one shared decomposition per (row × side) at committed layers.
    decomps = _side_decomps(sides, rows, ctx_ids, capture_layers, committed_ref, cc, pca_dim)
    k0_parity = (
        {"skipped_self_paired_smoke": True}
        if self_paired_k0
        else _empty_side_refit_parity(decomps, rows, committed_ref, args.committed_recon_he)
    )

    # (a) paired dose bootstrap per k (the imported round-4 machinery; the
    # shared index matrix is IDENTICAL across k — default_rng(seed) is
    # deterministic given (seed, B, n), asserted below).
    idx = np.random.default_rng(args.seed).integers(0, n, size=(args.n_boot, n))
    assert (idx == np.random.default_rng(args.seed).integers(0, n, size=(args.n_boot, n))).all()
    by_k: dict[str, dict] = {}
    cell_draws: dict[tuple[str, str], np.ndarray] = {}
    for k in ks:
        tag = f"k{btdr_pct(k)}"
        logger.info("[phase=paired] %s: full-side refit parity + shared-index bootstrap", tag)
        paired = _paired_bootstrap(
            args, full, sides[tag], rows, ctx_ids, capture_layers, committed_ref, cc
        )
        logger.info("[phase=mechanism] %s: centered cosine per (row × layer)", tag)
        mech = _mechanism_read(full, sides[tag], rows, ctx_ids, capture_layers, committed_ref)
        for r in rows:
            cell_draws[(r, str(btdr_pct(k)))] = np.asarray(paired["draws_by_row"][r])
        by_k[tag] = {
            "truncate_frac": k,
            "self_paired_smoke": k in self_paired_ks,
            **paired,
            "mechanism_by_row": mech,
        }

    fam27 = _familywise(cell_draws, "all rows × interior k")
    fam6 = _familywise(
        {c: d for c, d in cell_draws.items() if c[0] in PRIMARY_SINGLES},
        "primary singles (turn_nl, uh_im_start) × interior k",
    )
    if not args.smoke:
        assert fam27["n_cells"] == 27, fam27["n_cells"]
        assert fam6["n_cells"] == 6, fam6["n_cells"]

    side_names = ["full", "k0"] + [f"k{btdr_pct(k)}" for k in ks]
    abs_skill, per_ctx_skill = _abs_skill_and_percontext(decomps, rows, side_names, idx)
    per_ctx_delta = {
        s: {
            r: {
                c: (
                    None
                    if np.isnan(per_ctx_skill["full"][r][i]) or np.isnan(per_ctx_skill[s][r][i])
                    else float(per_ctx_skill["full"][r][i] - per_ctx_skill[s][r][i])
                )
                for i, c in enumerate(ctx_ids)
            }
            for r in rows
        }
        for s in side_names
        if s != "full"
    }

    k0_echo = _k0_committed_echo(args.committed_paired, rows, committed_ref)

    out_dir = Path(args.out)
    dump_json(
        {
            "dv": "paired_dose_response_delta_skill_full_minus_truncated",
            "method": (
                "per-context (ss_res, ss_tot) decompositions per (row, side) at each row's "
                "COMMITTED best layer (round-4 table), canonical manifest context order; "
                "ONE shared (B, n) context-resample index matrix (default_rng(seed), "
                "deterministic given (seed, B, n)) paired across EVERY (row × k × side) "
                "cell; full side refit-parity-asserted vs the committed round-1/round-3 "
                "skills ≤1e-6 per k; k=0 side refit-parity-asserted vs the committed "
                "round-4 skills ≤1e-6; k=0 paired deltas ECHOED from the committed "
                "round-4 JSON (never re-derived); verdicts per plan v18 §3 (equivalence "
                "margin ±0.02); mid-k absolute skills are descriptive trace points "
                "(NO band verdicts — registered)"
            ),
            "rows": rows,
            "truncate_fracs": ks,
            "n_contexts": n,
            "n_boot": args.n_boot,
            "seed": args.seed,
            "pca_dim": pca_dim,
            "committed_layers": {r: committed_ref[r][0] for r in rows},
            "self_paired_smoke_k0": self_paired_k0,
            "self_paired_smoke_ks": self_paired_ks,
            "empty_side_refit_parity": k0_parity,
            "k0_committed": k0_echo,
            "by_k": by_k,
            "familywise_27cell": fam27,
            "familywise_6cell_primary": fam6,
            "abs_skill_by_side": abs_skill,
            "per_context_delta_by_side": per_ctx_delta,
            "reproducibility": reproducibility_metadata(),
            "smoke": args.smoke,
        },
        out_dir / "paired_dose_response.json",
    )
    logger.info("[phase=paired] wrote %s", out_dir / "paired_dose_response.json")

    dump_json(
        {
            "dv": "cross_context_centered_similarity_full_vs_truncated_per_k",
            "rows": rows,
            "truncate_fracs": ks,
            "n_contexts": n,
            "capture_layers": capture_layers,
            "self_paired_smoke_ks": self_paired_ks,
            "by_k": {tag: by_k[tag]["mechanism_by_row"] for tag in by_k},
            "reproducibility": reproducibility_metadata(),
            "smoke": args.smoke,
        },
        out_dir / "mechanism_cosine_btdr.json",
    )
    logger.info("[phase=mechanism] wrote %s", out_dir / "mechanism_cosine_btdr.json")
    logger.info(
        "[phase=done] btdr dose driver complete (%d rows × %d k; fam27 p=%.4f, fam6 p=%.4f)",
        len(rows),
        len(ks),
        fam27["p_ge1_abs_centered_delta_ge_margin"],
        fam6["p_ge1_abs_centered_delta_ge_margin"],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
