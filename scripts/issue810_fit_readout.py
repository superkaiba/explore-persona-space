#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (→, ρ, ×, ², r_B) in scientific docstrings + log messages.
"""Issue #810 DV (b) — behavior read-out: held-out ρ of predicted vs graded E0.

Reads a behavior E0(C,B) out of the answer-side summary two ways:
- **method (i) fixed r_B** — the faithful #658 A3.3 projection ``E0 ≈ r_Bᵀ summary``
  (zero parameters); runs on the behaviors with an r_B in #658's store
  ({harmful_compliance, sycophancy, refusal}; broad_em dropped — floors on base).
- **method (ii) trained LOCO-ridge** — a learned ``summary → E0`` map (E0 as a
  1-column target) via ``vectorized_mlp_skill.ridge_predict_loco_centered``; runs
  on ALL graded-E0 behaviors.

PRIMARY target = the graded 0-100 E0 mean (llm-judging rule 1); COMPANION = the
binary judged-RATE (rule 2). Both are read per (context, behavior). High-m E0
comes from #810 Phase C (``e0_highm_graded.json``); low-m E0 from #763 (IN-FLIGHT
— OPTIONAL, folded when it lands; its absence is reported, never a crash).

NULLS (selection-symmetric, plan §6 + Alternatives-lens binding Concern):
- **fixed r_B** is a ZERO-parameter projection → the on-main permute-and-refit
  ridge null is WRONG for it. The correct null PERMUTES the (E0, summary) context
  pairing and RE-PROJECTS the SAME r_B (breaks alignment, preserves marginals).
- **trained ridge** null PERMUTES the E0 rows and RE-FITS the ridge.
Per-draw stats for EVERY (summary × layer × behavior × method) cell are persisted
to ``null_matrix_readout.json`` so the analyzer recomputes the honest
max-over-{summary, layer, behavior, method} band as a 0-GPU re-reduction.

H2 CONJUNCTION (Statistics + Methodology reconciler binding): a summary counts as
an H2 confirmation ONLY if it lifts read-out on BOTH methods (fixed r_B AND
trained ridge) for a behavior. Per-method lifts are REPORTED separately; the
conjunction gate is encoded (``h2_conjunction`` field per (behavior × summary)),
NOT a method-OR.

JUDGE-VALIDATION (llm-judging rules 13/15): Spearman(graded-E0, #722 tf-margin)
across the 50 contexts for {refusal, sycophancy} (the overlapping ± pool).
harmful_compliance has NO ± pool → validation gap NOTED, never fabricated. Reads
the tf_margin committed to main by §4.0 step 1.

LENGTH/VERBOSITY control (llm-judging rule 10, Alternatives-lens binding): per
behavior, correlate answer length with (a) the winning summary's read-out
prediction and (b) graded E0.

SELF-CONTAINED against on-main primitives (no import of the stranded
fig-per-position script).

Usage::

    uv run python scripts/issue810_fit_readout.py \\
        --e0-highm eval_results/issue_810/e0_highm_graded.json \\
        --position-store-hf issue658_theory_assumptions/answer_position_sweep \\
        --out eval_results/issue_810

    # smoke (1 behavior, 1 layer, fixed-r_B only): --smoke
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
from scipy.stats import spearmanr  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402
from issue810_batched_null import (  # noqa: E402
    batched_projection_null_rho,
    batched_ridge_loco_null_rho,
    make_perm_matrix,
)
from issue810_common import (  # noqa: E402
    BETLEY_E0_HIGHM_FILE,
    G1_ANSWER_POSITION_SWEEP_SUBDIR,
    G1_OUT_DIR,
    G1_STORE_MANIFEST,
    G1_V0_SUMMARIES,
    GENRES,
    HF_DATA_REPO,
    HF_PREFIX,
    I658_RB,
    I658_STORE_MANIFEST,
    I658_V0_SUMMARIES,
    I722_TF_MARGIN_FILE,
    PCA_TARGET_DIM_CAP,
    SHUFFLE_NULL_PERMS,
    SHUFFLE_NULL_SEED,
    TF_MARGIN_VALIDATION_BEHAVIORS,
    UH_SUMMARY_NAMES,
    assert_g1_probe_pool_hash,
    context_ids_from_manifest,
    dump_json,
    enlarged_summary_names,
    load_json,
    reproducibility_metadata,
    summary_names,
    upload_out_dir,
    validate_uh_pack,
)
from issue810_fit_reconstruction import _expand_rows  # noqa: E402

from explore_persona_space.analysis.vectorized_mlp_skill import (  # noqa: E402
    ridge_predict_loco_centered,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv(str(PROJECT_ROOT / ".env"))

logger = logging.getLogger("issue810_fit_readout")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def _rho(pred: np.ndarray, meas: np.ndarray) -> float | None:
    """Spearman ρ, guarded against degenerate (constant / tiny-n) inputs."""
    if len(pred) < 4 or np.std(pred) < 1e-9 or np.std(meas) < 1e-9:
        return None
    r, _ = spearmanr(pred, meas)
    return None if np.isnan(r) else float(r)


def _tf_margin_scalar(cell) -> float | None:
    """Extract the scalar tf-margin from a tf_margin cell.

    tf_margin schema: ``margins[ctx][behavior]`` is a DICT
    ``{"margin": <float>, "pos_mean_ln_logp", ...}`` — the scalar we validate
    against is ``["margin"]``. A bare-scalar or missing cell is handled too.
    """
    if isinstance(cell, dict):
        return cell.get("margin")
    return cell


# ── inputs ────────────────────────────────────────────────────────────────────


def _load_free_summaries(genre: str = "betley"):
    """{recipe: {ctx: (28,H)}} from the summaries-genre's v0_summaries.pt (g1 hash-pinned)."""
    from huggingface_hub import hf_hub_download

    v0_file = I658_V0_SUMMARIES if genre == "betley" else G1_V0_SUMMARIES
    p = hf_hub_download(HF_DATA_REPO, v0_file, repo_type="dataset")
    blob = torch.load(p, weights_only=False)
    if genre == "g1":
        assert_g1_probe_pool_hash(blob, G1_V0_SUMMARIES)
    return blob["summaries"], blob["capture_layers"]


def _load_rb():
    """{behavior: {recipe: (28,H)}} from #658 store/r_b.pt (recipes diffmeans/meanDB).

    PINNED to the parent's Betley ``store/r_b.pt`` in EVERY cell of the 2×2
    square — the co-located g1 store's ``r_b.pt`` is a DIFFERENT tensor
    (max|Δ| >= 1.0, plan-time verified) and swapping it would smuggle a second
    variable (plan v6 §11). Assert-fails if the resolved path carries the g1
    ``store_genre`` prefix (the plan §8 wrong-r_B risk).
    """
    from huggingface_hub import hf_hub_download

    if "store_genre" in I658_RB:
        raise RuntimeError(f"r_B constant drifted to a genre store: {I658_RB}")
    p = hf_hub_download(HF_DATA_REPO, I658_RB, repo_type="dataset")
    if "store_genre" in str(p):
        raise RuntimeError(
            f"r_B resolved to a genre store path ({p}) — the fixed direction must be the "
            "parent's Betley store/r_b.pt in every cell (plan v6 §11)"
        )
    blob = torch.load(p, weights_only=False)
    return blob["r_b"], blob.get("columns", list(blob["r_b"].keys()))


def _load_position_summaries(ctx_ids, hf_prefix, local_dir):
    from huggingface_hub import hf_hub_download

    out: dict[str, dict[str, np.ndarray]] = {}
    cov: dict[str, dict[str, int]] = {}
    for c in ctx_ids:
        if local_dir is not None and (local_dir / f"{c}.pt").is_file():
            blob = torch.load(local_dir / f"{c}.pt", weights_only=False)
        else:
            path = hf_hub_download(HF_DATA_REPO, f"{hf_prefix}/{c}.pt", repo_type="dataset")
            blob = torch.load(path, weights_only=False)
        pv = blob["pos_vectors"].float().numpy()  # (n_pos, Lc, H)
        out[c] = {name: pv[i] for i, name in enumerate(blob["positions"])}
        cov[c] = dict(blob["coverage"])
    return out, cov


def _e0_by_context(e0_highm: dict, low_m: dict | None) -> dict[str, dict[str, dict]]:
    """{behavior: {"graded": {ctx:val}, "rate": {ctx:val}}} merged high-m + low-m.

    High-m: #810 Phase C ``e0_highm_graded.json`` (graded mean + binary rate).
    Low-m: #763 graded E0 (OPTIONAL, IN-FLIGHT) — a dict {behavior: {ctx: score}}
    when landed, else None (its absence is reported by the caller, not a crash).
    """
    out: dict[str, dict[str, dict]] = {}
    for behavior, blk in e0_highm.get("by_behavior", {}).items():
        out[behavior] = {
            "graded": {k: v for k, v in blk["per_context_graded_mean"].items() if v is not None},
            "rate": {
                k: v for k, v in blk.get("per_context_binary_rate", {}).items() if v is not None
            },
        }
    if low_m:
        for behavior, ctx_scores in low_m.items():
            out.setdefault(behavior, {"graded": {}, "rate": {}})
            out[behavior]["graded"] = {k: v for k, v in ctx_scores.items() if v is not None}
    return out


def _load_uh_summaries(spec: str) -> tuple[dict, dict, dict]:
    """Load the compact uh_summaries pack (local path, else an HF data-repo path).

    Returns ``({row: {ctx: (Lc, H) fp32 np}}, {row: {ctx: probe count}}, meta)``
    — the 9 new-row source for the read-out enlarged-axis rerun (plan v11 §4.6
    item 5; avoids re-downloading the ~430 MB uh position store on the CPU
    chain). ``meta`` carries {smoke, context_ids, capture_layers, model}.
    Fails loud on a non-extended pack.
    """
    from huggingface_hub import hf_hub_download

    p = Path(spec)
    if not p.is_file():
        p = Path(hf_hub_download(HF_DATA_REPO, spec, repo_type="dataset"))
    blob = torch.load(p, weights_only=False)
    if not blob.get("extended_boundary"):
        raise RuntimeError(f"uh_summaries pack at {spec} lacks extended_boundary provenance")
    rows = {
        row: {c: t.float().numpy() for c, t in per_ctx.items()}
        for row, per_ctx in blob["summaries"].items()
    }
    meta = {
        k: blob.get(k)
        for k in (
            "smoke",
            "context_ids",
            "capture_layers",
            "model",
            "ablate_answer",
            "truncate_frac",  # `_btdr` per-k pack provenance (None on older packs)
            "rows",
        )
    }
    return rows, blob["coverage"], meta


def _summary_matrix(summary, layer_i, kept, free_summaries, pos_summaries, coverage, uh_rows=None):
    """(n, H) summary matrix at one layer over the kept ctx_ids (coverage-checked).

    Row sources: free recipes (mean/last/maxp) from v0; uh rows from the
    ``--uh-summaries`` pack when provided; every other position row from the
    position store. Parent behavior is byte-identical when ``uh_rows`` is None.
    """
    rows = []
    for c in kept:
        if summary in ("mean", "last", "maxp"):
            rows.append(free_summaries[summary][c][layer_i].numpy())
        elif uh_rows is not None and summary in uh_rows:
            rows.append(uh_rows[summary][c][layer_i])
        else:
            rows.append(pos_summaries[c][summary][layer_i])
    return np.stack(rows)


def _kept_contexts(summary, ctx_ids, coverage, uh_cov=None):
    """Contexts with coverage for this summary (free recipes always covered)."""
    if summary in ("mean", "last", "maxp"):
        return list(ctx_ids)
    if uh_cov is not None and summary in uh_cov:
        return [c for c in ctx_ids if uh_cov[summary].get(c, 0) > 0]
    return [c for c in ctx_ids if coverage[c].get(summary, 0) > 0]


def _default_behaviors(e0: dict, args) -> list[str]:
    """Behavior list with the plan-§5 quarantine applied to the default set.

    The PARENT's harmful-compliance E0 is cache-contaminated — using it anywhere
    but the contamination diagnostic is banned. The parent cell (betley, betley)
    is REUSED, never re-fit, so the filter fires only on the NEW square cell
    (g1 acts → parent E0); harmful compliance headlines only at E0_g1. An
    explicit ``--behaviors`` overrides (a deliberate diagnostic read).
    """
    behaviors = args.behaviors or list(e0.keys())
    if (
        args.behaviors is None
        and args.e0_genre == "betley"
        and args.summaries_genre != "betley"
        and "harmful_compliance" in behaviors
    ):
        behaviors = [b for b in behaviors if b != "harmful_compliance"]
        logger.info(
            "[phase=quarantine] harmful_compliance EXCLUDED from the (g1 acts -> parent E0) "
            "cell — the parent target is quarantined (plan v6 §5); it headlines only at E0_g1"
        )
    return behaviors


def _resolve_e0_path(e0_genre: str) -> Path:
    """Default graded-E0 JSON per E0-target genre (explicit --e0-highm overrides).

    betley → the parent's committed Phase-C output (branch issue-810; never
    re-judged this round). g1 → this round's Phase C-g output under the
    follow-up-label dir (written by ``issue810_batch_rejudge_highm.py --genre g1``).
    """
    if e0_genre == "betley":
        return BETLEY_E0_HIGHM_FILE
    return G1_OUT_DIR / "phase_c" / "e0_highm_graded.json"


# ── read-out methods ──────────────────────────────────────────────────────────


def _fixed_rb_pred(X: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Method (i): zero-parameter projection E0 ≈ r_Bᵀ summary (no fit)."""
    return X @ r


def _pca_reduce_predictor(X: np.ndarray) -> np.ndarray:
    """PCA-reduce the summary design to its top min(48, n-2) dims (fixed per cell).

    Shared by ``_trained_ridge_pred`` and the batched trained-ridge null so BOTH
    use the IDENTICAL basis — the design X is fixed across permutations, so the
    PCA basis is too; the batched null re-uses this reduced design rather than
    re-fitting the basis per permutation.
    """
    from explore_persona_space.analysis.vectorized_mlp_skill import robust_pca_basis

    n = X.shape[0]
    k = min(PCA_TARGET_DIM_CAP, max(1, n - 2))
    mu, comps, _ = robust_pca_basis(X, k)
    return (X - mu) @ comps.T  # (n, k) PCA-reduced predictor


def _trained_ridge_pred(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Method (ii): held-out LOCO-ridge prediction of scalar E0 from the summary.

    Uses the on-main ``ridge_predict_loco_centered`` with a 1-column target;
    reduces the summary to its top PCA dims first (target dim min(48, n-2)) so
    the H-dim predictor matches #722/#658's estimator. Returns (n,) held-out.
    """
    Xp = _pca_reduce_predictor(X)
    pred = ridge_predict_loco_centered(Xp, y.reshape(-1, 1))  # (n, 1)
    return pred[:, 0]


# ── main ──────────────────────────────────────────────────────────────────────


def _handle_rb_shape_mismatch(smoke, behavior, summary, layer_label, x_h, r_h, n_skips) -> int:
    """Fixed-r_B hidden-size mismatch: loud error in production, counted skip in smoke.

    A mixed-model SMOKE (0.5B store rows vs the 7B r_B) legitimately cannot
    project across hidden sizes — the skip is logged + counted (persisted as
    ``fixed_rb_shape_skips`` in the output JSON). Production shapes always
    match, so a NON-smoke mismatch means the summaries store and r_B come from
    different models — raise, never a silent skip (r1 Minor: the bare
    ``continue`` was undocumented + uncounted).
    """
    if not smoke:
        raise RuntimeError(
            f"fixed_rb hidden-size mismatch on a NON-smoke run: summary rows H={x_h} vs "
            f"r_B H={r_h} ({behavior}/{summary}/L{layer_label}) — the summaries store "
            "and r_B come from different models"
        )
    n_skips += 1
    logger.info(
        "[phase=skip] fixed_rb %s/%s L%s: hidden-size mismatch (%d vs %d) — "
        "mixed-model smoke skip #%d",
        behavior,
        summary,
        layer_label,
        x_h,
        r_h,
        n_skips,
    )
    return n_skips


def _resolve_rows_and_sources(args, pos_man: dict, ctx_ids: list[str], capture_layers: list[int]):
    """Expand --rows, load + validate the uh pack, fail loud on missing sources.

    Returns ``(summaries, uh_rows, uh_cov)``. A requested uh row with NO source
    (neither the ``--uh-summaries`` pack nor an extended-boundary position
    store) refuses — never a silent KeyError mid-fit. On a NON-smoke run the
    loaded pack is validated against the production grid BEFORE the fit loop
    (``validate_uh_pack``: non-smoke provenance, model, the full layer axis,
    every requested row × every production context — r1 CONCERN
    ``uh-pack-meta-validation-readout``); a ``--smoke`` run keeps the relaxed
    path (partial coverage pairs via ``_kept_contexts``). ``--null-mode
    full-rerun`` additionally requires the FULL 46-row × 28-layer axis (unless
    --smoke; plan v11 §6 read mode 1 — the enlarged-axis band's denominator).
    """
    uh_rows, uh_cov = (None, None)
    uh_meta: dict = {}
    if args.uh_summaries:
        uh_rows, uh_cov, uh_meta = _load_uh_summaries(args.uh_summaries)
        logger.info("[phase=load] uh_summaries pack: %d rows", len(uh_rows))
    summaries = _expand_rows(args.summaries) if args.summaries else summary_names()
    uh_requested = [s for s in summaries if s in set(UH_SUMMARY_NAMES)]
    if uh_requested and uh_rows is None and not pos_man.get("extended_boundary"):
        raise SystemExit(
            f"rows {uh_requested} requested but no --uh-summaries pack given and the "
            "position store is not extended-boundary — no source for the new rows"
        )
    if uh_rows is not None:
        if args.smoke:
            logger.info(
                "[phase=load] --smoke: uh pack production validation RELAXED "
                "(smoke=%s model=%s) — partial coverage pairs via _kept_contexts",
                uh_meta.get("smoke"),
                uh_meta.get("model"),
            )
        else:
            validate_uh_pack(
                uh_rows,
                uh_cov,
                uh_meta,
                requested_rows=uh_requested,
                ctx_ids=ctx_ids,
                expected_capture_layers=capture_layers,
            )
            logger.info(
                "[phase=load] uh pack VALIDATED: %d requested rows x %d contexts x %d layers",
                len(uh_requested),
                len(ctx_ids),
                len(capture_layers),
            )
    if args.null_mode == "full-rerun" and not args.smoke:
        missing = sorted(set(enlarged_summary_names()) - set(summaries))
        if missing or (args.layers is not None):
            raise SystemExit(
                "--null-mode full-rerun requires the FULL 46-row x 28-layer axis "
                f"(missing rows: {missing[:5]}...; layers subset: {args.layers}) — "
                "pass --rows all-46 with no --layers (plan v11 §6 read mode 1)"
            )
    return summaries, uh_rows, uh_cov


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #810 DV (b): behavior read-out rho")
    ap.add_argument(
        "--summaries-genre",
        choices=list(GENRES),
        default="betley",
        help="genre of the ACTIVATION side (v0 summaries + position store): 'betley' "
        "(default — the parent's sources, bit-for-bit) or 'g1' (#658's UltraChat arm)",
    )
    ap.add_argument(
        "--e0-genre",
        choices=list(GENRES),
        default="betley",
        help="genre of the E0 TARGET side: 'betley' (the parent's committed Phase-C "
        "graded E0) or 'g1' (this round's Phase C-g output). The 2x2 square = "
        "--summaries-genre x --e0-genre (plan v6 §4.6 item 4)",
    )
    ap.add_argument(
        "--e0-highm",
        default=None,
        help="explicit graded-E0 JSON path (overrides the --e0-genre default resolution)",
    )
    ap.add_argument("--e0-lowm", default=None, help="#763 graded E0 JSON (optional; in-flight)")
    ap.add_argument(
        "--position-store-hf",
        default=None,
        help="HF prefix of the aligned-subset position store (default: the "
        "--summaries-genre's — answer_position_sweep[_<genre-tag>])",
    )
    ap.add_argument("--position-store-dir", default=None)
    ap.add_argument(
        "--out",
        default=None,
        help="output dir (default: eval_results/issue_810 when BOTH genres are betley — "
        "the parent path, bit-for-bit — else the follow-up round dir "
        "eval_results/issue_810/ultrachat-genre-summary-sweep so a g1 run never "
        "clobbers the parent's committed JSONs)",
    )
    ap.add_argument(
        "--summaries",
        "--rows",
        nargs="*",
        default=None,
        help="subset of summary rows (default = the parent 37-row set). Accepts the tokens "
        "'uh-new' (the 9 new rows) and 'all-46' (the enlarged axis). `--rows` is the "
        "plan-v11 alias.",
    )
    ap.add_argument("--layers", nargs="*", type=int, default=None)
    ap.add_argument("--behaviors", nargs="*", default=None)
    ap.add_argument("--methods", nargs="*", default=["fixed_rb", "trained_ridge"])
    ap.add_argument(
        "--uh-summaries",
        default=None,
        help="uh_summaries.pt pack (local path or HF data-repo path) sourcing the 9 new "
        "rows — the CPU-chain input; without it, uh rows resolve from the position "
        "store (which must then be the extended-boundary store)",
    )
    ap.add_argument(
        "--null-mode",
        choices=["per-run", "full-rerun"],
        default="per-run",
        help="'per-run' = parent behavior byte-for-bit (per-cell nulls for whatever was "
        "fit). 'full-rerun' = the plan-v11 read-out enlarged-axis primary path: nulls "
        "recomputed for EVERY fitted cell (NO --null-join exists on this leg — the "
        "shared-rng stream makes a join invalid, A5 fact-check) + the enlarged-axis "
        "max-selected band + the per-behavior two-method conjunction statistic emitted; "
        "requires the fitted rows to cover the full 46-row axis (unless --smoke).",
    )
    ap.add_argument(
        "--out-suffix",
        default=None,
        help="output filename tag: readout_rho_<suffix>.json + null_matrix_readout_"
        "<suffix>.json (default None = the parent filenames, byte-for-bit)",
    )
    ap.add_argument("--n-perms", type=int, default=SHUFFLE_NULL_PERMS)
    ap.add_argument(
        "--device",
        default="cpu",
        help="torch device for the batched trained-ridge + projection nulls ('cpu' default, "
        "'cuda' to run on a GPU lane — CPU behavior is byte-identical to the default)",
    )
    ap.add_argument(
        "--upload-prefix",
        default=None,
        help="HF data-repo path prefix (e.g. 'issue810/phase_d_readout') to bulk-upload the "
        "out-dir *.json to after the fit; UNSET (default) = no upload (today's behavior). "
        "Set on an ephemeral GCP lane so the results survive instance teardown.",
    )
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    # Fail fast: --device cuda with no CUDA never silently falls back to CPU.
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is False")

    from huggingface_hub import hf_hub_download

    any_g1 = "g1" in (args.summaries_genre, args.e0_genre)
    out_dir = Path(
        args.out or (G1_OUT_DIR if any_g1 else (PROJECT_ROOT / "eval_results" / "issue_810"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.position_store_hf is None:
        args.position_store_hf = (
            f"{HF_PREFIX}/answer_position_sweep"
            if args.summaries_genre == "betley"
            else f"{HF_PREFIX}/{G1_ANSWER_POSITION_SWEEP_SUBDIR}"
        )
    e0_path = Path(args.e0_highm) if args.e0_highm else _resolve_e0_path(args.e0_genre)

    logger.info(
        "[phase=load] manifest + summaries + r_B + E0 + tf_margin "
        "(summaries_genre=%s e0_genre=%s e0=%s)",
        args.summaries_genre,
        args.e0_genre,
        e0_path,
    )
    manifest_file = I658_STORE_MANIFEST if args.summaries_genre == "betley" else G1_STORE_MANIFEST
    man = load_json(hf_hub_download(HF_DATA_REPO, manifest_file, repo_type="dataset"))
    if args.summaries_genre == "g1":
        assert_g1_probe_pool_hash(man, G1_STORE_MANIFEST)
    ctx_ids_all = context_ids_from_manifest(man)
    free_summaries, capture_layers = _load_free_summaries(args.summaries_genre)
    rb, _rb_columns = _load_rb()

    local_dir = Path(args.position_store_dir) if args.position_store_dir else None
    if local_dir is not None:
        pos_man = load_json(local_dir / "manifest.json")
    else:
        pos_man = load_json(
            hf_hub_download(
                HF_DATA_REPO, f"{args.position_store_hf}/manifest.json", repo_type="dataset"
            )
        )
    ctx_ids = [c for c in pos_man["context_ids"] if c in ctx_ids_all]
    pos_summaries, coverage = _load_position_summaries(ctx_ids, args.position_store_hf, local_dir)

    e0_highm = load_json(e0_path)
    low_m = load_json(args.e0_lowm) if args.e0_lowm and Path(args.e0_lowm).is_file() else None
    if args.e0_lowm and low_m is None:
        logger.warning(
            "low-m E0 (%s) NOT landed — read-out folds high-m only (in-flight dep)", args.e0_lowm
        )
    e0 = _e0_by_context(e0_highm, low_m)

    summaries, uh_rows, uh_cov = _resolve_rows_and_sources(args, pos_man, ctx_ids, capture_layers)
    layers = args.layers if args.layers is not None else list(range(len(capture_layers)))
    behaviors = _default_behaviors(e0, args)
    rng = np.random.default_rng(SHUFFLE_NULL_SEED)

    results: list[dict] = []
    # null_matrix[behavior][method][summary][layer] = [per-draw ρ]
    null_matrix: dict = {}
    fixed_rb_shape_skips = 0

    for behavior in behaviors:
        graded = e0.get(behavior, {}).get("graded", {})
        rates = e0.get(behavior, {}).get("rate", {})
        if len(graded) < 4:
            logger.info("[phase=skip] %s: <4 graded contexts (%d) — skipped", behavior, len(graded))
            continue
        null_matrix.setdefault(behavior, {})
        for method in args.methods:
            if method == "fixed_rb" and behavior not in rb:
                logger.info(
                    "[phase=skip] fixed_rb has no r_B for %s — trained_ridge only", behavior
                )
                continue
            null_matrix[behavior].setdefault(method, {})
            for summary in summaries:
                kept = [
                    c for c in _kept_contexts(summary, ctx_ids, coverage, uh_cov) if c in graded
                ]
                if len(kept) < 4:
                    continue
                y = np.array([graded[c] for c in kept], dtype=np.float64)
                y_rate = np.array(
                    [rates.get(c, np.nan) for c in kept], dtype=np.float64
                )  # companion
                null_matrix[behavior][method].setdefault(summary, {})
                for li in layers:
                    X = _summary_matrix(
                        summary, li, kept, free_summaries, pos_summaries, coverage, uh_rows
                    )
                    # Draw n_perms permutations from the SHARED rng in the SAME
                    # order the serial per-draw loop consumed them (byte-identical
                    # null on a like-seeded rng — the smoke asserts this).
                    perm = make_perm_matrix(len(kept), args.n_perms, rng)
                    if method == "fixed_rb":
                        # diffmeans is the theory default; report both recipes but
                        # gate the headline on diffmeans (persona-vectors default).
                        r = rb[behavior]["diffmeans"][li].numpy()
                        if X.shape[1] != r.shape[0]:
                            fixed_rb_shape_skips = _handle_rb_shape_mismatch(
                                args.smoke,
                                behavior,
                                summary,
                                capture_layers[li],
                                X.shape[1],
                                r.shape[0],
                                fixed_rb_shape_skips,
                            )
                            continue
                        pred = _fixed_rb_pred(X, r)
                        # correct null (batched, no re-fit): permute the (E0,
                        # summary) pairing, re-project the SAME pred → re-Spearman.
                        draws = batched_projection_null_rho(pred, y, perm, device=args.device)
                    else:  # trained_ridge
                        pred = _trained_ridge_pred(X, y)
                        # null (batched): permute E0 rows, re-fit the LOCO ridge on
                        # the FIXED PCA-reduced design (X-only factors computed once,
                        # all perms batched — the #722 vectorize mandate). Uses the
                        # IDENTICAL PCA basis as _trained_ridge_pred.
                        Xp = _pca_reduce_predictor(X)
                        draws = batched_ridge_loco_null_rho(Xp, y, perm, device=args.device)
                    rho = _rho(pred, y)
                    rho_rate = _rho(pred, y_rate) if np.isfinite(y_rate).all() else None
                    results.append(
                        {
                            "behavior": behavior,
                            "method": method,
                            "summary": summary,
                            "layer": capture_layers[li],
                            "n": len(kept),
                            "rho_graded": rho,  # PRIMARY
                            "rho_binary_rate": rho_rate,  # COMPANION
                        }
                    )
                    null_matrix[behavior][method][summary][str(capture_layers[li])] = draws
            logger.info("[phase=fit] %s / %s done", behavior, method)

    # H2 conjunction: per (behavior × summary), does it lift on BOTH methods vs
    # the `mean` summary baseline? Recorded as a gate the analyzer reads (the
    # honest band re-reduction happens post-hoc from null_matrix).
    conjunction = _h2_conjunction(results)

    # Judge validation: Spearman(graded-E0, tf-margin) for {refusal, sycophancy}.
    judge_val = _judge_validation(e0, ctx_ids)

    # Length/verbosity control: answer length vs (winning-summary pred, graded E0).
    length_control = _length_control(
        results,
        e0,
        ctx_ids,
        pos_man,
        free_summaries,
        pos_summaries,
        coverage,
        layers,
        capture_layers,
        rb,
        uh_rows,
        uh_cov,
    )

    _write_outputs(
        args,
        out_dir,
        e0_path,
        ctx_ids,
        results,
        null_matrix,
        conjunction,
        judge_val,
        length_control,
        low_m,
        fixed_rb_shape_skips,
    )
    logger.info("[phase=done] wrote read-out results + null matrix to %s", out_dir)
    return 0


def _write_outputs(
    args,
    out_dir: Path,
    e0_path,
    ctx_ids,
    results,
    null_matrix,
    conjunction,
    judge_val,
    length_control,
    low_m,
    fixed_rb_shape_skips: int = 0,
) -> None:
    """Persist the read-out results + null matrix (+ the full-rerun reductions).

    Enlarged-axis reductions (plan v11 §6 / H4-uh; --null-mode full-rerun): the
    max-selected band over EVERY fitted cell's freshly recomputed draws + the
    per-behavior two-method conjunction statistic (max over summaries of min
    over methods of best-layer ρ, identical selection per draw — the g1
    conjunction_bands recipe) recomputed over the enlarged summary axis.
    ``--out-suffix`` retargets the filenames so a uh run never clobbers the
    parent's committed JSONs.
    """
    enlarged = None
    if args.null_mode == "full-rerun":
        enlarged = _enlarged_axis_reductions(results, null_matrix, args.n_perms)

    readout_name = (
        f"readout_rho_{args.out_suffix}.json" if args.out_suffix else "readout_rho_by_summary.json"
    )
    null_name = (
        f"null_matrix_readout_{args.out_suffix}.json"
        if args.out_suffix
        else "null_matrix_readout.json"
    )
    dump_json(
        {
            "dv": "behavior_readout_rho_vs_graded_e0",
            "summaries_genre": args.summaries_genre,
            "e0_genre": args.e0_genre,
            "e0_source": str(e0_path),
            "primary": "rho_graded (graded 0-100 E0)",
            "companion": "rho_binary_rate (judged rate >=50)",
            "n_contexts_grid": len(ctx_ids),
            "behaviors_fit": sorted({r["behavior"] for r in results}),
            "methods": args.methods,
            "null_mode": args.null_mode,
            "low_m_e0_landed": low_m is not None,
            "fixed_rb_shape_skips": fixed_rb_shape_skips,  # >0 on mixed-model smoke only
            "cells": results,
            "h2_conjunction": conjunction,
            "enlarged_axis": enlarged,
            "judge_validation": judge_val,
            "length_control": length_control,
            "reproducibility": reproducibility_metadata(),
            "smoke": args.smoke,
        },
        out_dir / readout_name,
    )
    dump_json(
        {
            "dv": "readout",
            "summaries_genre": args.summaries_genre,
            "e0_genre": args.e0_genre,
            "axes": "behavior -> method -> summary -> layer -> [per-draw rho]",
            "n_perms": args.n_perms,
            "seed": SHUFFLE_NULL_SEED,
            "null_mode": args.null_mode,
            "readout": null_matrix,
        },
        out_dir / null_name,
    )
    if args.upload_prefix:
        logger.info("[phase=upload] fit-result JSONs -> %s", args.upload_prefix)
        landed = upload_out_dir(out_dir, args.upload_prefix)
        logger.info("[phase=upload] verified fit-result JSONs under %s/", landed)


def _conjunction_reductions(results: list[dict], null_matrix: dict, n_perms: int) -> dict:
    """Per-behavior two-method conjunction statistic + its max-selected band.

    Statistic (the g1 ``conjunction_bands`` recipe, recomputed over THIS run's
    summary axis): max over summaries of min over methods of best-layer ρ.
    Null: the IDENTICAL selection applied to each per-draw matrix (best over
    layers per draw → min over methods → max over summaries). A summary
    lacking either method (no r_B / mixed-model skip) is excluded from BOTH
    the observed and the null reduction (selection symmetry preserved).
    """
    methods = sorted({r["method"] for r in results})
    out: dict[str, dict] = {}
    if len(methods) < 2:
        return {"note": f"conjunction needs 2 methods; got {methods}", "by_behavior": out}
    best: dict[tuple, float] = {}
    for r in results:
        if r["rho_graded"] is None:
            continue
        key = (r["behavior"], r["summary"], r["method"])
        best[key] = max(best.get(key, -2.0), r["rho_graded"])
    for beh in sorted({b for (b, _s, _m) in best}):
        per_summary: dict[str, float] = {}
        for s in {s for (b, s, _m) in best if b == beh}:
            vals = [best.get((beh, s, m)) for m in methods]
            if any(v is None for v in vals):
                continue
            per_summary[s] = min(vals)
        if not per_summary:
            continue
        arg = max(per_summary, key=lambda k: per_summary[k])
        names = sorted(per_summary)
        mins = None
        computable = True
        for m in methods:
            per_s = null_matrix.get(beh, {}).get(m, {})
            mats = []
            for s in names:
                layer_draws = [
                    np.asarray(d, dtype=np.float64)
                    for d in per_s.get(s, {}).values()
                    if len(d) == n_perms
                ]
                if not layer_draws:
                    computable = False
                    break
                mats.append(np.stack(layer_draws).max(axis=0))  # best-over-layers per draw
            if not computable:
                break
            stacked = np.stack(mats)  # (S, draws)
            mins = stacked if mins is None else np.minimum(mins, stacked)
        band = None
        if computable and mins is not None:
            conj_draws = mins.max(axis=0)  # (draws,)
            band = float(np.percentile(conj_draws, 97.5))
        stat = per_summary[arg]
        out[beh] = {
            "statistic": stat,
            "arg_summary": arg,
            "band_97_5": band,
            "verdict": (
                "not_computable"
                if band is None
                else ("clears_band" if stat > band else "within_band")
            ),
            "per_summary": per_summary,
        }
    return {"methods": methods, "by_behavior": out}


def _enlarged_axis_reductions(results: list[dict], null_matrix: dict, n_perms: int) -> dict:
    """Enlarged-axis max-selected band + conjunction (plan v11 §6, full-rerun mode).

    Band: per-draw max over EVERY fitted (behavior × method × summary × layer)
    cell's freshly recomputed null draws (the read-out leg NEVER joins — the
    shared-rng stream bars it, A5) → 97.5th percentile, vs the best observed ρ.
    """
    all_draws = [
        np.asarray(draws, dtype=np.float64)
        for per_m in null_matrix.values()
        for per_s in per_m.values()
        for per_l in per_s.values()
        for draws in per_l.values()
        if len(draws) == n_perms
    ]
    band = None
    if all_draws:
        band = float(np.percentile(np.stack(all_draws).max(axis=0), 97.5))
    obs_cells = [
        (r["behavior"], r["method"], r["summary"], r["layer"], r["rho_graded"])
        for r in results
        if r["rho_graded"] is not None
    ]
    obs = max((c[4] for c in obs_cells), default=None)
    obs_arg = max(obs_cells, key=lambda c: c[4]) if obs_cells else None
    verdict = (
        "not_computable"
        if (obs is None or band is None)
        else ("clears_band" if obs > band else "within_band")
    )
    return {
        "max_selected": {
            "statistic": obs,
            "arg_cell": list(obs_arg) if obs_arg else None,
            "band_97_5": band,
            "n_cells": len(all_draws),
            "verdict": verdict,
        },
        "conjunction": _conjunction_reductions(results, null_matrix, n_perms),
    }


def _h2_conjunction(results: list[dict]) -> dict:
    """Per (behavior × summary): does it beat the `mean`-summary ρ on BOTH methods?

    H2 CONJUNCTION reading (binding). Reports the best-layer ρ per (behavior ×
    summary × method) and the `mean`-summary best-layer ρ per (behavior ×
    method), and flags ``both_methods_lift`` iff the summary's best-layer ρ
    exceeds the `mean` baseline's best-layer ρ on BOTH methods present for the
    behavior. This is the encoded H2 gate; the honest max-selected band comes
    from null_matrix post-hoc (analyzer).
    """
    by = {}  # (behavior, method, summary) -> best-layer rho
    methods_by_behavior: dict[str, set] = {}
    for r in results:
        if r["rho_graded"] is None:
            continue
        key = (r["behavior"], r["method"], r["summary"])
        by[key] = max(by.get(key, -2.0), r["rho_graded"])
        methods_by_behavior.setdefault(r["behavior"], set()).add(r["method"])
    out = {}
    behaviors = {k[0] for k in by}
    for behavior in behaviors:
        methods = sorted(methods_by_behavior.get(behavior, set()))
        summaries = {k[2] for k in by if k[0] == behavior}
        mean_rho = {m: by.get((behavior, m, "mean")) for m in methods}
        out[behavior] = {"methods": methods, "mean_baseline_best_rho": mean_rho, "summaries": {}}
        for s in summaries:
            if s == "mean":
                continue
            per_method = {m: by.get((behavior, m, s)) for m in methods}
            lifts = {
                m: (
                    per_method[m] is not None
                    and mean_rho[m] is not None
                    and per_method[m] > mean_rho[m]
                )
                for m in methods
            }
            out[behavior]["summaries"][s] = {
                "best_rho_per_method": per_method,
                "lifts_per_method": lifts,
                "both_methods_lift": len(methods) >= 2 and all(lifts.values()),
            }
    return out


def _judge_validation(e0: dict, ctx_ids: list[str]) -> dict:
    """Spearman(graded-E0, #722 tf-margin) for {refusal, sycophancy}.

    tf_margin schema: {margins: {ctx: {behavior: <margin>}}, behaviors: [...]}.
    harmful_compliance has NO ± pool -> gap noted, never fabricated. Reads the
    tf_margin committed to main by §4.0 step 1 (git-tree path, resolved locally
    on any lane that cloned the up-to-date branch/main).
    """
    tf_path = PROJECT_ROOT / I722_TF_MARGIN_FILE
    if not tf_path.is_file():
        return {"available": False, "reason": f"tf_margin not present at {tf_path}"}
    tf = load_json(tf_path)
    margins = tf["margins"]  # {ctx: {behavior: margin}}
    tf_behaviors = set(tf.get("behaviors", []))
    out = {"available": True, "tf_behaviors": sorted(tf_behaviors), "by_behavior": {}}
    for behavior in TF_MARGIN_VALIDATION_BEHAVIORS:
        if behavior not in tf_behaviors:
            out["by_behavior"][behavior] = {"status": "no_tf_margin_pool"}
            continue
        graded = e0.get(behavior, {}).get("graded", {})
        pairs = []
        for c in ctx_ids:
            if c not in graded:
                continue
            tfv = _tf_margin_scalar(margins.get(c, {}).get(behavior))
            if tfv is not None:
                pairs.append((graded[c], tfv))
        if len(pairs) < 4:
            out["by_behavior"][behavior] = {"status": "insufficient_overlap", "n": len(pairs)}
            continue
        g = np.array([p[0] for p in pairs])
        m = np.array([p[1] for p in pairs])
        out["by_behavior"][behavior] = {"status": "computed", "n": len(pairs), "rho": _rho(g, m)}
    out["gap_note"] = "harmful_compliance has NO tf_margin ± pool — validation gap (not fabricated)"
    return out


def _length_control(
    results,
    e0,
    ctx_ids,
    pos_man,
    free_summaries,
    pos_summaries,
    coverage,
    layers,
    capture_layers,
    rb,
    uh_rows=None,
    uh_cov=None,
) -> dict:
    """Per behavior: corr(answer length, winning-summary read-out pred) + corr(len, graded E0).

    A length-explained lift is NOT a persona finding (llm-judging rule 10). Answer
    length is the per-context median answer token count (from the Phase B manifest's
    per_context_diag). Uses the winning (behavior, method, summary, layer) by best ρ.
    A winning UH row resolves through the uh pack's OWN coverage + row source
    (``_kept_contexts`` / ``_summary_matrix(..., uh_rows)``) — the r1 gap filtered
    UH rows through the position store's coverage, degrading every UH winner to
    ``insufficient`` (CONCERN ``uh-length-control-uh-winner-gap``).
    """
    per_ctx_len = {}
    for c, d in pos_man.get("per_context_diag", {}).items():
        per_ctx_len[c] = d.get("median_answer_len")
    out = {}
    # winning cell per behavior (max graded ρ across all method/summary/layer)
    best: dict[str, dict] = {}
    for r in results:
        if r["rho_graded"] is None:
            continue
        b = r["behavior"]
        if b not in best or r["rho_graded"] > best[b]["rho_graded"]:
            best[b] = r
    for behavior, cell in best.items():
        graded = e0.get(behavior, {}).get("graded", {})
        summary, li_val, method = cell["summary"], cell["layer"], cell["method"]
        li = capture_layers.index(li_val)
        covered = set(_kept_contexts(summary, ctx_ids, coverage, uh_cov))
        kept = [
            c
            for c in ctx_ids
            if c in covered and c in graded and c in per_ctx_len and per_ctx_len[c] is not None
        ]
        if len(kept) < 4:
            out[behavior] = {"status": "insufficient", "n": len(kept)}
            continue
        X = _summary_matrix(summary, li, kept, free_summaries, pos_summaries, coverage, uh_rows)
        if method == "fixed_rb" and behavior in rb:
            pred = X @ rb[behavior]["diffmeans"][li].numpy()
        else:
            pred = _trained_ridge_pred(X, np.array([graded[c] for c in kept]))
        lengths = np.array([per_ctx_len[c] for c in kept], dtype=np.float64)
        g = np.array([graded[c] for c in kept], dtype=np.float64)
        out[behavior] = {
            "winning_cell": {"method": method, "summary": summary, "layer": li_val},
            "n": len(kept),
            "rho_len_vs_pred": _rho(lengths, pred),
            "rho_len_vs_graded_e0": _rho(lengths, g),
        }
    return out


if __name__ == "__main__":
    raise SystemExit(main())
