#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (Δ, M⁺, M0, →, ρ, ×, Ŵ, ‖·‖, ※) in scientific docstrings + log messages.
"""Issue #813 — DVs: Δ/floor + chain-ρ + substrate-swap null + pairwise-diff CIs.

Runs OFF-POD (VM CPU) — closed-form ridge + sampling stats over the reduced
per-(behavior, substrate) summaries the extraction wave wrote. NOT an iterative
fit (ridge-only, ``include_mlp=False``), so it belongs on the CPU/VM.

Per (behavior, substrate) at the FROZEN headline layer 14 (#651/#658; applied
IDENTICALLY to the observed statistic AND the substrate-swap null — the
selection-symmetric frozen-position route, `.claude/rules/selection-symmetric-nulls.md`):

- em / fact / sycophancy → ``issue722_fit_M.fit_cell(behavior, 14, cells, rb_main,
  rb_fact, include_mlp=False)`` (the RIDGE-only headline: ``Delta_med`` /
  ``floor_combined`` / ``Delta_over_floor_sd`` / chain-ρ / support_distance /
  n_with_E). fact/syco/em have an r_B; marker does NOT (fit_cell KeyErrors on it).
- marker → ``issue667_marker_mapchange.fit_marker_layer(14, cells, wu_marker,
  with_chain=)`` (read-1 unprojected ‖ΔM‖/floor + read-2 W_U[※]-projected
  |ΔM·Ŵ_U[※]|/floor + ``wu_frac_in_subspace`` — read-2 uninformative when < 0.1).

Then, per behavior:
- **Substrate-swap null (matched-n).** Within EACH substrate, resample the
  substrate's questions and re-split them into TWO pseudo-substrates of the SAME n
  per pseudo-arm (question-average per context → a pseudo-map pair), compute the
  SAME Δ/floor DV for each pseudo-substrate, take |Δ/floor(A) − Δ/floor(B)|. The
  null holds ΔM fixed (same adapter) and varies ONLY the question sample — its
  95th percentile is the behavior-specific threshold X a REAL substrate difference
  must clear (plan §3). Matched-n keeps em's low power conservative (not inflated).
- **Pairwise substrate-difference CIs (D1).** Family-clustered bootstrap CI on the
  SIGNED Δ/floor difference Δ/floor(A) − Δ/floor(B) for the three substrate pairs.
  Both substrates fit over the SAME shared 50-context battery (plan §4.2), so ONE
  family-clustered context resample refits BOTH arms per draw and the paired
  difference is recomputed each resample — the same refit machinery the observed
  read + the substrate-swap null use. The CI EXCLUDES 0 iff its whole interval is
  on one side of 0; that is the SECOND conjunct of the plan §3 verdict.
- **Verdict (D1 CONJUNCTION).** "substrate matters" (H0) iff BOTH conjuncts fire:
  (i) the max-vs-min Δ/floor difference exceeds the substrate-swap null p95 AND
  (ii) a DRIVING-pair pairwise CI (the pair whose difference IS max_diff) excludes
  0. "substrate-agnostic" (H1) iff BOTH fail (max within band AND all CIs include
  0). AMBIGUOUS (None) iff exactly one conjunct fires or a conjunct is undecidable.
  The reducer ``decide_substrate_matters`` is a pure function (unit-testable).

Reads the frozen headline-layer per-question rows (``per_question_L14.npz``) for
the null; reads the 28-layer reduced summary for the observed fit_cell/marker read
AND the pairwise-CI refit (both share the shared battery contexts per substrate).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# BEFORE the numpy/torch imports: the shared-VM thread-cap setdefaults (#847) bind
# in-process only when load_dotenv() runs before torch freezes its pools.
load_dotenv()

import issue658_fit_predictors as fit658  # noqa: E402
import issue667_marker_mapchange as marker_mc  # noqa: E402
import issue722_fit_M as fitM  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from issue722_bootstrap import floor_sd, make_refit_pair  # noqa: E402
from issue813_save_maps import _require_npz_keys  # noqa: E402  (fail-loud NPZ preflight)

logger = logging.getLogger("issue813.analysis")

DATA_REPO = "superkaiba1/explore-persona-space-data"
EXPERIMENT_NAME = "issue813_mapchange_substrate"
BEHAVIORS = ("em", "fact", "sycophancy", "marker")
SUBSTRATES = ("generic", "elicit", "mix")
HEADLINE_LAYER = 14  # frozen (#651/#658); observed + null read at the SAME layer
HIDDEN = 3584
N_LAYERS = 28
TARGET_DIM = 64  # top-64 v0 PCs (NEVER 48)
N_NULL_RESAMPLES = 1000
NULL_SEED = 42
# Refit-pair count for the PER-PSEUDO-ARM floor inside the substrate-swap null (B2).
# The observed read uses 100 (issue722_fit_M.N_REFIT_PAIRS); the null refits a floor
# per pseudo-arm per resample (n_resamples × 2 arms × NULL_REFIT_PAIRS refits), so it
# uses fewer pairs to stay tractable — each pseudo-floor is a coarse-but-honest
# per-arm estimate, and the null's own resampling dominates the band width. Smoke
# clamps this via --null-refit-pairs.
NULL_REFIT_PAIRS = 40
# The plan §3 "substrate matters" decision rule (D1): a CONJUNCTION of the null-band
# gate AND a driving-pair pairwise-CI excluding 0 — NOT the single null-band gate the
# round-1/2 verdict shipped. One constant so the reducer + the summary metadata agree.
_DECISION_RULE = "conjunction: (max_diff > null_p95) AND (a driving-pair pairwise CI excludes 0)"


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return "unknown"


# ── Per-cell resume-skip (boundary-swap enabler) ───────────────────────────────


def _load_cell_json(path: Path) -> dict | None:
    """Load + JSON-parse one cell artifact; return None on missing / corrupt / non-dict.

    A missing, truncated, or non-object file yields None so the caller RECOMPUTES the
    cell (overwriting the partial write) — never a silent half-loaded state.
    """
    if not path.exists():
        return None
    try:
        obj = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _null_regime_matches(null: dict, n_resamples: int, n_refit_pairs: int) -> bool:
    """True iff the loaded null was produced under the CURRENT output-affecting regime.

    Keys the resume decision on EVERY output-affecting arg the null band depends on —
    ``--n-null-resamples`` and ``--null-refit-pairs`` (#722-r3: a resume that ignores an
    output-affecting arg silently reuses wrong cached rows). A stale smoke/debug null
    (fewer draws or a different refit-pair count) is REJECTED so the cell recomputes
    rather than corrupting summary.json under the current run's metadata.

    Degenerate nulls (``note`` present AND ``n_over_floor_resamples_used == 0``) encode a
    STRUCTURAL property of the cell (too few questions / every resample degenerate) — no
    usable band was produced. Their handling splits on whether the null is STAMPED with the
    regime it was produced under (``n_null_resamples_requested`` present):

    - A STAMPED degenerate must pass the SAME exact-match regime checks as a full-success
      null (n_refit_pairs exact AND n_null_resamples_requested exact). ``issue813_dispatch.sh
      --smoke`` writes into the SAME ``eval_results/issue_813`` out-dir at ``max_questions=2``,
      which produces "too few questions" degenerate nulls — and post-r3 those carry the
      regime stamp, so a production resume MUST regime-check them and RECOMPUTE the smoke
      degenerate rather than preserve a smoke ``note`` JSON under production metadata (Codex
      round-2 Major blocker: smoke-in-shared-out-dir hole). Post-fix smokes always stamp, so
      every FUTURE smoke degenerate is regime-checked here.
    - An UNSTAMPED degenerate keeps the vacuous bypass. Those are LEGACY production artifacts
      that predate the regime stamp (a real too-few-questions cell from before r3): there is
      no scale to compare, no refits contributed, and the cell produced no usable band, so the
      regime checks are not comparable quantities for it and it stays resumable.

    ``n_refit_pairs`` check: the loaded value must equal ``n_refit_pairs``; a null MISSING
    that key (a legacy full-success artifact predating the regime stamp) FAILS the check
    (recompute) — refit pairs materially shape the floor.

    Requested-resamples check: if ``n_null_resamples_requested`` is stamped, it must equal
    ``n_resamples``. If UNSTAMPED (a legacy production full-success artifact — e.g. the
    em/generic null the live run is writing right now), accept iff
    ``n_resamples_used >= ceil(0.9 * n_resamples)`` — a production-scale null loses <10%
    of its draws to degenerate resamples, while a smoke-scale null (requested 5-50) can
    never clear the floor against a production ``n_resamples`` (e.g. 1000).
    """
    is_degenerate = "note" in null and null.get("n_over_floor_resamples_used", 0) == 0
    # A STAMPED degenerate (written post-r3, carries n_null_resamples_requested) falls through
    # to the SAME exact-match regime checks below as a full-success null — so a smoke degenerate
    # (max_questions=2) sharing the production out-dir is REJECTED, not preserved. Only an
    # UNSTAMPED (legacy, pre-stamp) degenerate keeps the vacuous bypass.
    if is_degenerate and "n_null_resamples_requested" not in null:
        return True
    # refit-pairs must match exactly (a legacy full-success null missing the key fails).
    if null.get("n_refit_pairs") != n_refit_pairs:
        return False
    # requested-resamples: stamped ⇒ exact match; unstamped legacy ⇒ >=90%-of-current used.
    requested = null.get("n_null_resamples_requested")
    if requested is not None:
        return requested == n_resamples
    used = null.get("n_resamples_used", 0)
    return used >= math.ceil(0.9 * n_resamples)


def _resume_cell(
    behavior: str,
    substrate: str,
    delta_dir: Path,
    null_dir: Path,
    n_resamples: int,
    n_refit_pairs: int,
) -> tuple[dict, dict] | None:
    """Return (obs, null) loaded from disk iff BOTH cell JSONs are complete AND match the
    current output-affecting regime, else None.

    A cell is RESUMABLE iff BOTH its ``delta_floor`` (observed read) JSON AND its
    ``substrate_swap_null`` JSON exist AND parse as a JSON object AND the null JSON
    carries the completion signal — either ``n_over_floor_resamples_used >= 1`` (the
    full-success shape) OR a ``note`` field (the degenerate-cell early-return shape,
    e.g. "too few questions" / "all resamples degenerate") — AND the loaded null matches
    the current ``n_resamples`` / ``n_refit_pairs`` regime (``_null_regime_matches``). A
    regime MISMATCH (a stale smoke/debug null with fewer draws or a different refit-pair
    count) is REJECTED so the cell recomputes rather than being silently reused under the
    current run's summary metadata (#722-r3 output-affecting-key discipline). Any missing /
    corrupt / truncated file OR an in-flight null with neither completion signal ⇒ None
    (recompute the cell from scratch, overwriting the partial files). The returned
    ``obs`` / ``null`` dicts are the EXACT structures ``observed_read`` /
    ``substrate_swap_null`` produce (the writers use ``json.dumps(default=float)``, so JSON
    round-trip float coercion is the only difference — accepted by the downstream
    pairwise/verdict/summary consumers).
    """
    obs = _load_cell_json(delta_dir / f"{behavior}__{substrate}.json")
    null = _load_cell_json(null_dir / f"{behavior}__{substrate}.json")
    if obs is None or null is None:
        return None
    null_complete = null.get("n_over_floor_resamples_used", 0) >= 1 or "note" in null
    if not null_complete:
        return None
    if not _null_regime_matches(null, n_resamples, n_refit_pairs):
        return None
    return obs, null


# ── Observed Δ/floor read (reused fit machinery) ───────────────────────────────


def _cells_from_summary(behavior: str, substrate: str, layer: int, reduced_root: Path) -> list:
    """Build CellRecord list at ``layer`` from the reduced summary (import shared loader)."""
    import issue813_save_maps as savemaps813

    return savemaps813.load_reduced_cells(behavior, substrate, layer, reduced_root)


def observed_read(
    behavior: str,
    substrate: str,
    reduced_root: Path,
    rb_main: dict,
    rb_fact: dict | None,
    wu_marker: np.ndarray | None,
) -> dict:
    """The observed Δ/floor read at the frozen headline layer for one (behavior, substrate)."""
    cells = _cells_from_summary(behavior, substrate, HEADLINE_LAYER, reduced_root)
    if behavior == "marker":
        # marker has no r_B → the two-read marker path (unproj ‖ΔM‖ + W_U[※]-proj).
        cell = marker_mc.fit_marker_layer(
            HEADLINE_LAYER, cells, wu_marker, with_chain=(substrate != "generic")
        )
        return {
            "behavior": behavior,
            "substrate": substrate,
            "layer": HEADLINE_LAYER,
            "n_cells": cell["n_cells"],
            # read-1 (behavior-agnostic) is the marker's PRIMARY floor-normalized DV
            "delta_over_floor": cell["unproj_delta_over_floor"],
            "delta_over_floor_sd": cell["unproj_delta_over_floor_sd"],
            "delta_med": cell["unproj_delta_med"],
            "floor_combined": cell["unproj_floor_p95"]["combined"],
            # read-2 (W_U[※]-projected, marker-specific) + its subspace-capture gate
            "wu_delta_over_floor": cell["wu_proj_delta_over_floor"],
            "wu_frac_in_subspace": cell["wu_frac_in_subspace"],
            "wu_read2_informative": cell["wu_read2_informative"],
            "support_distance": cell["support_distance"],
            "chain_rho": cell.get("chain_rho"),
            "marker_two_read": True,
        }
    # em / fact / sycophancy → the ridge-only headline (Delta_med / floor / chain-ρ).
    cell = fitM.fit_cell(behavior, HEADLINE_LAYER, cells, rb_main, rb_fact, include_mlp=False)
    return {
        "behavior": behavior,
        "substrate": substrate,
        "layer": HEADLINE_LAYER,
        "n_cells": cell["n_cells"],
        "delta_over_floor": cell["Delta_over_floor_sd"],
        "delta_over_floor_sd": cell["Delta_over_floor_sd"],
        "delta_med": cell["Delta_med"],
        "floor_combined": cell["floor_combined"],
        "support_distance": cell["support_distance"],
        "chain_rho": cell["chain_rho"],
        "n_with_E": cell["chain_rho"].get("n_with_E"),
        "marker_two_read": False,
    }


# ── Pseudo-substrate Δ/floor read (headline-layer only, for the null) ──────────


def _pseudo_delta_over_floor(
    c0: np.ndarray,
    cplus: np.ndarray,
    v0: np.ndarray,
    vplus: np.ndarray,
    families: list[str],
    r_hat: np.ndarray | None,
    *,
    n_refit_pairs: int,
) -> tuple[float, float]:
    """(Δ_med, Δ/floor) for a headline-layer pseudo-map — B2: the REGISTERED DV space.

    Fits M0 = ridge(c0→V0_64) and M⁺ = ridge(cplus→Vplus_64) at THIS layer via the
    reused ``_ridge_fit_predict`` + ``_pca_basis_v0`` (top-64 shared V0 basis) and
    reduces the base-grid difference by the r_hat projection (em/fact/syco r_B) or by
    the vector norm (marker read-1) — EXACTLY ``fit_cell`` / ``fit_marker_layer``'s
    numerator. It ALSO refits a per-pseudo-arm FLOOR through the SAME shared harness
    (``make_refit_pair`` for the r_hat path / ``marker_mc._refit_pair_norm`` for the
    norm path, over the M0 / M⁺ / shifted refit designs, family-clustered) and returns
    the normalized DV in each behavior's own convention (em/fact/syco: ``Δ_med /
    floor_sd_combined`` matching ``Delta_over_floor_sd``; marker: ``Δ_med /
    floor_p95_combined`` matching ``unproj_delta_over_floor``), so the null band is
    built in the REGISTERED Δ/floor space, not raw Δ (concern
    i813-verdict-raw-delta-not-registered-floor). Returns ``(delta_med, delta_over_floor)``;
    ``delta_over_floor`` is NaN when the floor underflows (excluded by the caller).
    """
    pca_basis = fitM._pca_basis_v0(v0, TARGET_DIM)  # (k<=64, HIDDEN)
    v0_64 = fitM._to64(v0, pca_basis)
    vplus_64 = fitM._to64(vplus, pca_basis)
    m0_grid = fitM._ridge_fit_predict(c0, v0_64, c0)  # (n, 64)
    mplus_grid = fitM._ridge_fit_predict(cplus, vplus_64, c0)
    delta_full = (mplus_grid - m0_grid) @ pca_basis  # (n, HIDDEN)
    if r_hat is None:
        delta_med = float(np.median(np.linalg.norm(delta_full, axis=1)))
    else:
        delta_med = float(np.median(np.abs(delta_full @ r_hat)))

    # Per-pseudo-arm refit floor via the SHARED harness (same three refit designs the
    # observed read uses: M0, M⁺, shifted M0(cplus)). Grid = the base c0, matching the
    # numerator's eval grid. r_hat=None routes through the marker read-1 ‖·‖ variant.
    fit_fn = fitM._refit_ridge_fn(c0)  # returns preds at c0, back-projected to HIDDEN
    m0_at_cplus = fitM.m0_at_cplus_ridge_full(c0, v0, cplus, pca_basis)
    if r_hat is None:
        fl_m0 = marker_mc._refit_pair_norm(c0, v0, fit_fn, c0, families, n_pairs=n_refit_pairs)
        fl_mp = marker_mc._refit_pair_norm(
            cplus, vplus, fit_fn, c0, families, n_pairs=n_refit_pairs
        )
        fl_sh = marker_mc._refit_pair_norm(
            cplus, m0_at_cplus, fit_fn, c0, families, n_pairs=n_refit_pairs
        )
        # marker read-1 normalizes by the p95-COMBINED floor (unproj_delta_over_floor).
        floor = max(
            float(np.percentile(fl_m0, 95)),
            float(np.percentile(fl_mp, 95)),
            float(np.percentile(fl_sh, 95)),
        )
    else:
        fl_m0 = make_refit_pair(c0, v0, fit_fn, c0, r_hat, families, n_pairs=n_refit_pairs)
        fl_mp = make_refit_pair(cplus, vplus, fit_fn, c0, r_hat, families, n_pairs=n_refit_pairs)
        fl_sh = make_refit_pair(
            cplus, m0_at_cplus, fit_fn, c0, r_hat, families, n_pairs=n_refit_pairs
        )
        # em/fact/syco normalize by the SD-COMBINED floor (Delta_over_floor_sd).
        floor = max(floor_sd(fl_m0), floor_sd(fl_mp), floor_sd(fl_sh))
    dof = float("nan") if floor < 1e-12 else float(delta_med / floor)
    return delta_med, dof


# ── Batched substrate-swap-null engine (Gram/dual-space; B3 vectorization) ──────
#
# The serial battery costs ~2-3.6h/cell: n_resamples × 2 pseudo-arms × (2 observed
# ridge fits + 3 floors × n_refit_pairs pairs × 2 refits) ≈ 488k tiny fits, each a
# fresh numpy→torch round-trip + a handful of (n≈50, HIDDEN) GEMMs — the
# `.claude/rules/vectorize-many-cell-fits.md` perm/bootstrap-battery pattern
# (overhead-bound, not FLOP-bound). The batched engine reproduces the SAME math
# with the fit axis batched:
#
# - The Y side (PCA basis + Y64 + r̂/‖·‖ projections) never touches HIDDEN space:
#   the per-fit basis is computed from the (row × row) Gram of the arm-level
#   stacks (eigh of the double-centered Gram == the SVD route on the non-null
#   spectrum), and every downstream product (Y64, pca@r̂, cross-basis Grams for
#   the marker ‖·‖ read) reduces to gathers of arm-level (n, n) Grams.
# - The X side (per-fit column standardization) is irreducibly HIDDEN-dim (each
#   fit divides by its OWN per-column sd), so each fit standardizes the arm's
#   combined [C0; C⁺] stack once and takes ONE batched (2n, HIDDEN) GEMM; the
#   per-fit design Gram + grid cross-terms are then gathers of that (2n, 2n)
#   weighted Gram.
# - Ridge PRESS / λ-selection / dual solve run batched in zero-padded row space:
#   padding is EXACT for the dual ridge (padded rows carry zero design + zero
#   target ⇒ zero dual coefficients) and leaves the PRESS argmin unchanged
#   (padding rescales the mean LOO-MSE by a λ-independent constant).
#
# Fidelity tier (b) DISTRIBUTIONAL, not bit-exact: the serial `_pca_basis_v0`
# keeps k = min(dim, rows) SVD rows even when the mean-centered stack has rank
# < k (always true: rank ≤ distinct_rows − 1), so the serial tail basis rows are
# numerically arbitrary null-space directions (gesdd noise scaled by 1/s at
# s ≈ 0). The batched Gram route TRUNCATES those null directions (relative
# eigenvalue threshold `_EIG_TRUNC_REL`) instead of reproducing gesdd's
# arbitrary ones — the non-null (real-signal) basis matches the serial SVD to
# float tolerance; only the serial tail noise differs. The rng draw SEQUENCES
# (question resamples, family-clustered floor resamples) are reproduced exactly.

_EIG_TRUNC_REL = 1e-10  # relative eigenvalue (s²) cutoff for null-space truncation


def _resolve_null_device(requested: str) -> str:
    """'auto' → cuda when available else cpu; explicit 'cpu'/'cuda' pass through."""
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _serial_battery_tombstone(name: str) -> None:
    """Tombstone guard for the SUPERSEDED serial battery (Supersede contract,
    `.claude/rules/vectorize-many-cell-fits.md`): warn loudly, and refuse under
    EPM_FORBID_SERIAL_FITS=1 so a follow-up round cannot silently re-run it."""
    if os.environ.get("EPM_FORBID_SERIAL_FITS") == "1":
        raise RuntimeError(
            f"{name}: the serial per-fit battery is superseded by the batched Gram-space "
            "engine (the default). EPM_FORBID_SERIAL_FITS=1 forbids the serial path; drop "
            "--serial-null (or unset the env) to proceed."
        )
    warnings.warn(
        f"{name}: running the SUPERSEDED serial battery (tombstoned twin, kept only as the "
        "--serial-null verification escape hatch; the batched Gram-space engine is the "
        "default and ~50-100x faster)",
        FutureWarning,
        stacklevel=3,
    )


def _floor_resample_indices(
    fams: list[str], n_pairs: int, seed: int = 0
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Replicate ``make_refit_pair`` / ``_refit_pair_norm``'s rng consumption EXACTLY.

    Returns the ``n_pairs`` (idx_a, idx_b) row-index arrays the serial harness draws
    with ``default_rng(seed=0)`` — the family-clustered double resample (or the iid
    fallback for <2 families), INCLUDING the two per-pair ``rng.integers`` draws the
    serial code spends on refit-init seeds (the ridge fit ignores them, but they
    advance the stream). The sequence depends only on (fams, n_pairs, seed), so it
    is shared across every arm with the same kept-family signature AND across the
    three floor targets (each serial floor call constructs its own default_rng(0)).
    """
    n = len(fams)
    fams_arr = np.asarray(list(fams), dtype=object)
    uniq = sorted({str(f) for f in fams_arr})
    clustered = len(uniq) >= 2
    fam_to_idx = {f: np.where(fams_arr.astype(str) == f)[0] for f in uniq}
    rng = np.random.default_rng(seed)
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for _ in range(n_pairs):
        if clustered:
            chosen_a = rng.choice(uniq, size=len(uniq), replace=True)
            idx_a = np.concatenate([fam_to_idx[str(f)] for f in chosen_a])
            chosen_b = rng.choice(uniq, size=len(uniq), replace=True)
            idx_b = np.concatenate([fam_to_idx[str(f)] for f in chosen_b])
        else:
            idx_a = rng.integers(0, n, size=n)
            idx_b = rng.integers(0, n, size=n)
        rng.integers(0, 2**31 - 1)  # the serial pair's rng_a init seed (discarded here)
        rng.integers(0, 2**31 - 1)  # rng_b
        out.append((np.asarray(idx_a, dtype=np.int64), np.asarray(idx_b, dtype=np.int64)))
    return out


def _gather_sub(G: torch.Tensor, idx_r: torch.Tensor, idx_c: torch.Tensor) -> torch.Tensor:
    """Sub-matrix gather: G (B, n, n), idx_r (B, mr), idx_c (B, mc) → (B, mr, mc)."""
    n = G.shape[-1]
    rows = torch.gather(G, 1, idx_r.unsqueeze(-1).expand(-1, -1, n))
    return torch.gather(rows, 2, idx_c.unsqueeze(1).expand(-1, rows.shape[1], -1))


def _masked_median(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """np.median semantics (average the two middle order stats) along the LAST dim."""
    big = torch.finfo(x.dtype).max
    xm = torch.where(mask, x, torch.full_like(x, big))
    s, _ = torch.sort(xm, dim=-1)
    cnt = mask.sum(dim=-1)
    lo = ((cnt - 1) // 2).clamp(min=0)
    hi = (cnt // 2).clamp(min=0)
    a = torch.gather(s, -1, lo.unsqueeze(-1)).squeeze(-1)
    b = torch.gather(s, -1, hi.unsqueeze(-1)).squeeze(-1)
    return (a + b) / 2.0


def _right_center(Gs: torch.Tensor, wcol: torch.Tensor, colmask: torch.Tensor) -> torch.Tensor:
    """[Yb Ycᵀ]_{ij} = y_i·(y_j − μ) from the plain product Gram (center the 2nd factor).

    Gs (B, m, m') = Yb Yᵀ; wcol (B, m') = mean weights over the CENTERED factor's real
    rows (sum 1); colmask (B, m') zeroes padded columns. Padded Yb rows are already
    zero rows of Gs, so they stay zero.
    """
    rc = Gs - torch.einsum("bij,bj->bi", Gs, wcol).unsqueeze(-1)
    return rc * colmask.to(rc.dtype).unsqueeze(1)


def _gram_pca_coeff(
    Gy_sub: torch.Tensor,
    imask: torch.Tensor,
    m_real: torch.Tensor,
    k_cap: torch.Tensor,
    kmax: int,
) -> torch.Tensor:
    """Row-coefficient PCA basis from a masked/padded Y Gram (the `_pca_basis_v0` twin).

    Gy_sub (B, m, m) = Yb Ybᵀ over the resampled rows (padded rows/cols zero); returns
    C (B, kmax, m) with pca_basis == C @ Yc (row-coefficient form): the top-k
    right-singular-vector basis of the mean-centered Yc via eigh of the
    double-centered Gram. Rows beyond min(k_cap, numerical rank) are ZERO
    (truncated) — the tier-(b) deviation from gesdd's arbitrary null-space tails.
    """
    w = imask.to(Gy_sub.dtype) / m_real.clamp(min=1.0).unsqueeze(-1)
    rm = torch.einsum("bij,bj->bi", Gy_sub, w)  # row means == col means (Gs symmetric)
    gm = torch.einsum("bi,bi->b", rm, w)
    Gc = Gy_sub - rm.unsqueeze(-1) - rm.unsqueeze(-2) + gm.unsqueeze(-1).unsqueeze(-1)
    mask2 = imask.unsqueeze(-1) & imask.unsqueeze(-2)
    Gc = torch.where(mask2, Gc, torch.zeros_like(Gc))
    evals, U = torch.linalg.eigh(Gc)  # ascending
    evals = evals.flip(-1)[:, :kmax]  # (B, kmax) top eigenvalues (== s²)
    U = U.flip(-1)[:, :, :kmax]  # (B, m, kmax)
    ev_max = evals[:, :1].clamp(min=0.0)
    j = torch.arange(kmax, device=evals.device).unsqueeze(0)
    valid = (j < k_cap.unsqueeze(-1)) & (evals > _EIG_TRUNC_REL * ev_max) & (evals > 0)
    inv_s = torch.where(valid, evals.clamp(min=1e-300).rsqrt(), torch.zeros_like(evals))
    C = inv_s.unsqueeze(-1) * U.transpose(-1, -2)  # (B, kmax, m)
    return C * imask.to(C.dtype).unsqueeze(1)


def _batched_press_ridge(
    Gd: torch.Tensor, Y: torch.Tensor, Ks: list[torch.Tensor], lambdas: list[float]
) -> list[torch.Tensor]:
    """Batched PRESS λ-selection + dual-ridge predictions (padded rows exact-zero).

    Mirrors ``fit658._press_loo_mse_per_lambda`` + ``_ridge_dual_weights``: ONE eigh
    of the standardized design Gram Gd (B, m, m) reused across the λ grid; per-fit
    argmin λ; alpha = Q diag(1/(ev+λ*)) Qᵀ Y; pred = K @ alpha per grid cross-Gram K
    (B, g, m). Zero-padded rows are exact: they carry zero design + zero target, so
    alpha_pad = 0 and the PRESS mean is rescaled by a λ-independent constant
    (argmin unchanged).
    """
    evals, Q = torch.linalg.eigh(Gd)
    QtY = Q.transpose(-1, -2) @ Y
    Qsq = Q * Q
    mses = torch.empty((Gd.shape[0], len(lambdas)), dtype=Gd.dtype, device=Gd.device)
    for li, lam in enumerate(lambdas):
        filt = evals / (evals + lam)
        h_diag = torch.einsum("bmj,bj->bm", Qsq, filt)
        Yhat = Q @ (filt.unsqueeze(-1) * QtY)
        resid = Y - Yhat
        denom = (1.0 - h_diag).clamp(min=1e-8).unsqueeze(-1)
        loo = resid / denom
        mses[:, li] = (loo * loo).mean(dim=(-1, -2))
    lam_t = torch.tensor(lambdas, dtype=Gd.dtype, device=Gd.device)
    lam_sel = lam_t[torch.argmin(mses, dim=1)]
    alpha = Q @ (QtY / (evals + lam_sel.unsqueeze(-1)).unsqueeze(-1))
    return [K @ alpha for K in Ks]


def _batched_arm_dofs(
    arms: list[dict],
    r_hat: np.ndarray | None,
    *,
    n_refit_pairs: int,
    device: str,
    pair_chunk: int = 4,
) -> list[tuple[float, float] | None]:
    """(Δ_med, Δ/floor) for a CHUNK of pseudo-arms — the batched `_pseudo_delta_over_floor`.

    ``arms``: dicts with c0/cplus/v0/vplus ((m_i, HIDDEN) float64 arrays) + fams
    (list[str], one label per row). Returns one (delta_med, dof) tuple per arm (dof
    NaN where the floor underflows, matching the serial 1e-12 gate), or None where
    the arm's linear algebra degenerated — the serial LinAlgError-skip analog. On a
    batched linear-algebra failure the whole chunk FALLS BACK to the serial
    reference per arm (exact serial semantics, incl. per-arm skips).
    """
    try:
        return _batched_arm_dofs_inner(
            arms, r_hat, n_refit_pairs=n_refit_pairs, device=device, pair_chunk=pair_chunk
        )
    except (RuntimeError, np.linalg.LinAlgError) as e:
        logger.warning(
            "[phase=analysis] batched arm engine failed on a %d-arm chunk (%s); "
            "falling back to the serial reference for this chunk",
            len(arms),
            e,
        )
        out: list[tuple[float, float] | None] = []
        for a in arms:
            try:
                out.append(
                    _pseudo_delta_over_floor(
                        a["c0"],
                        a["cplus"],
                        a["v0"],
                        a["vplus"],
                        a["fams"],
                        r_hat,
                        n_refit_pairs=n_refit_pairs,
                    )
                )
            except np.linalg.LinAlgError:
                out.append(None)
        return out


def _floor_design_gram(
    src: torch.Tensor,
    rows: torch.Tensor,
    rows_mask: torch.Tensor,
    off: int,
    armrep: torch.Tensor,
    cnt: torch.Tensor,
    idxf: torch.Tensor,
    imskf: torch.Tensor,
    gm2: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """(Gd, K) for one batch of resampled floor designs.

    Standardizes `rows` (the arm stacks the design + grid rows live in) by each
    fit's count-weighted moments of `src`, takes ONE batched HIDDEN-dim Gram, then
    gathers the design sub-Gram (Gd) and the grid (C0 rows) × design cross-terms
    (K). `off` places the design block inside `rows` (0 = C0 block, n = C⁺ block).
    """
    dt = src.dtype
    src_b = src[armrep]  # (B, n, H)
    mu = torch.einsum("bn,bnh->bh", cnt, src_b)
    e2 = torch.einsum("bn,bnh->bh", cnt, src_b * src_b)
    sd = (e2 - mu * mu).clamp(min=0).sqrt() + 1e-9
    z = (rows[armrep] - mu.unsqueeze(1)) / sd.unsqueeze(1)
    z = z * rows_mask[armrep].to(dt).unsqueeze(-1)
    gw = z @ z.transpose(1, 2)  # the ONE HIDDEN-dim GEMM per fit
    didx = idxf + off
    gd = torch.where(gm2, _gather_sub(gw, didx, didx), torch.zeros((), dtype=dt, device=src.device))
    k_cross = torch.gather(gw[:, :n, :], 2, didx.unsqueeze(1).expand(-1, n, -1)) * imskf.to(
        dt
    ).unsqueeze(1)  # (B, n, mF) grid × design
    return gd, k_cross


def _batched_arm_dofs_inner(
    arms: list[dict],
    r_hat: np.ndarray | None,
    *,
    n_refit_pairs: int,
    device: str,
    pair_chunk: int,
) -> list[tuple[float, float] | None]:
    """The batched engine body (see the section comment above for the math)."""
    dev = torch.device(device)
    dt = torch.float64
    n_arms = len(arms)
    n = max(a["c0"].shape[0] for a in arms)
    h = arms[0]["c0"].shape[1]

    def _pad(key: str) -> torch.Tensor:
        out = torch.zeros((n_arms, n, h), dtype=dt, device=dev)
        for i, a in enumerate(arms):
            out[i, : a[key].shape[0]] = torch.from_numpy(np.ascontiguousarray(a[key])).to(
                device=dev, dtype=dt
            )
        return out

    stack_c0, stack_cp, stack_v0, stack_vp = _pad("c0"), _pad("cplus"), _pad("v0"), _pad("vplus")
    m = torch.tensor([a["c0"].shape[0] for a in arms], dtype=torch.long, device=dev)
    mask = torch.arange(n, device=dev).unsqueeze(0) < m.unsqueeze(1)  # (A, n)
    mf = mask.to(dt)
    w_full = mf / m.to(dt).unsqueeze(1)

    # Floor resample indices, shared per kept-family signature + across floor targets.
    sig_cache: dict[tuple, list[tuple[np.ndarray, np.ndarray]]] = {}
    per_arm_pairs = []
    m_fit_max = 1
    for a in arms:
        sig = tuple(a["fams"])
        if sig not in sig_cache:
            sig_cache[sig] = _floor_resample_indices(list(sig), n_refit_pairs)
        pairs = sig_cache[sig]
        per_arm_pairs.append(pairs)
        m_fit_max = max(m_fit_max, max(max(len(ia), len(ib)) for ia, ib in pairs))
    fit_idx = torch.zeros((n_arms, n_refit_pairs, 2, m_fit_max), dtype=torch.long, device=dev)
    fit_msk = torch.zeros((n_arms, n_refit_pairs, 2, m_fit_max), dtype=torch.bool, device=dev)
    for ai, pairs in enumerate(per_arm_pairs):
        for pi, (ia, ib) in enumerate(pairs):
            fit_idx[ai, pi, 0, : len(ia)] = torch.from_numpy(ia).to(dev)
            fit_msk[ai, pi, 0, : len(ia)] = True
            fit_idx[ai, pi, 1, : len(ib)] = torch.from_numpy(ib).to(dev)
            fit_msk[ai, pi, 1, : len(ib)] = True

    lambdas = [float(x) for x in fit658.RIDGE_LAMBDAS]

    # ── Observed pseudo-arm read (the fit_cell/fit_marker_layer numerator) ──
    gram_v0 = stack_v0 @ stack_v0.transpose(1, 2)  # (A, n, n)
    gram_vp = stack_vp @ stack_vp.transpose(1, 2)
    gram_vpv0 = stack_vp @ stack_v0.transpose(1, 2)
    obs_dim = TARGET_DIM  # the 813 module global (the serial _pseudo path uses the same)
    obs_cap = torch.minimum(torch.full_like(m, obs_dim), m)
    k_obs = int(min(obs_dim, n))
    c_obs = _gram_pca_coeff(gram_v0, mask, m.to(dt), obs_cap, k_obs)  # (A, k_obs, n)
    y64_v0 = _right_center(gram_v0, w_full, mask) @ c_obs.transpose(1, 2)  # (A, n, k)
    y64_vp = _right_center(gram_vpv0, w_full, mask) @ c_obs.transpose(1, 2)

    comb = torch.cat([stack_c0, stack_cp], dim=1)  # (A, 2n, H)
    mask2n = torch.cat([mask, mask], dim=1)

    def _obs_gram(src: torch.Tensor) -> torch.Tensor:
        """Combined [C0; C⁺] Gram standardized by the FULL-stack moments of `src`."""
        mu = torch.einsum("an,anh->ah", w_full, src)
        e2 = torch.einsum("an,anh->ah", w_full, src * src)
        sd = (e2 - mu * mu).clamp(min=0).sqrt() + 1e-9
        z = (comb - mu.unsqueeze(1)) / sd.unsqueeze(1)
        z = z * mask2n.to(dt).unsqueeze(-1)
        return z @ z.transpose(1, 2)  # (A, 2n, 2n)

    g0 = _obs_gram(stack_c0)
    m064, m0cp64 = _batched_press_ridge(
        g0[:, :n, :n], y64_v0, [g0[:, :n, :n], g0[:, n:, :n]], lambdas
    )
    g1 = _obs_gram(stack_cp)
    (mplus64,) = _batched_press_ridge(g1[:, n:, n:], y64_vp, [g1[:, :n, n:]], lambdas)
    delta64 = mplus64 - m064  # (A, n, k_obs)
    if r_hat is not None:
        r_t = torch.from_numpy(np.ascontiguousarray(r_hat)).to(device=dev, dtype=dt)
        v0r = stack_v0 @ r_t  # (A, n)
        vpr = stack_vp @ r_t
        v0rc = (v0r - (w_full * v0r).sum(1, keepdim=True)) * mf
        q_obs = torch.einsum("akn,an->ak", c_obs, v0rc)  # pca_obs @ r̂ in coeff space
        proj = torch.einsum("ank,ak->an", delta64, q_obs).abs()
    else:
        # ‖d64 @ pca‖ == ‖d64‖ (orthonormal basis rows; truncated rows are zero).
        proj = delta64.norm(dim=-1)
    delta_med = _masked_median(proj, mask)  # (A,)

    # Shifted target M0(C⁺) lives in obs-basis coefficients — its Gram + r̂ read are
    # basis-free (orthonormal rows), so the fl_sh floor needs no HIDDEN-dim work.
    gram_sh = m0cp64 @ m0cp64.transpose(1, 2)  # (A, n, n)
    gy_by_t = [gram_v0, gram_vp, gram_sh]
    if r_hat is not None:
        yr_by_t = [v0r, vpr, torch.einsum("ank,ak->an", m0cp64, q_obs)]

    # ── The three refit floors, batched over (arm × pair × member) ──
    #
    # FLOP-sharing across floor targets: fl_mp and fl_sh resample the SAME design
    # (the C⁺ rows at the same indices — only their Y differs), so the design side
    # (standardization + the HIDDEN-dim Gram + PRESS inputs) is computed ONCE per
    # variant, not once per floor; and fl_m0's grid IS its design block (C0), so its
    # weighted Gram only needs the n-row C0 block, not the 2n combined one. Together
    # ~2.4x fewer HIDDEN-dim GEMM FLOPs than the naive one-Gram-per-floor shape.
    refit_dim = fitM.TARGET_DIM  # the serial _refit_ridge_fn reads issue722_fit_M's global
    k_ref = int(min(refit_dim, m_fit_max))
    stats = torch.zeros((n_arms, 3, n_refit_pairs), dtype=dt, device=dev)
    for p0 in range(0, n_refit_pairs, pair_chunk):
        pc = list(range(p0, min(p0 + pair_chunk, n_refit_pairs)))
        n_p = len(pc)
        idx = fit_idx[:, pc]  # (A, P, 2, mF)
        imsk = fit_msk[:, pc]
        bsz = n_arms * n_p * 2
        idxf = idx.reshape(bsz, m_fit_max)
        imskf = imsk.reshape(bsz, m_fit_max)
        m_real = imskf.sum(-1).to(dt)
        wf = imskf.to(dt) / m_real.clamp(min=1.0).unsqueeze(-1)
        cnt = torch.zeros((bsz, n), dtype=dt, device=dev)
        cnt.scatter_add_(1, idxf, wf)  # per-arm-row weights (multiplicity/m)
        armrep = torch.arange(n_arms, device=dev).repeat_interleave(n_p * 2)
        gm2 = imskf.unsqueeze(-1) & imskf.unsqueeze(-2)
        cap = torch.minimum(m_real.long(), torch.full_like(m_real.long(), refit_dim))

        # variant 0: X = C0 rows; grid == design block → the n-row C0 Gram suffices.
        # variant 1: X = C⁺ rows (fl_mp AND fl_sh — identical design, different Y);
        #            grid = C0 rows → the combined [C0; C⁺] Gram.
        design_by_var = (
            _floor_design_gram(stack_c0, stack_c0, mask, 0, armrep, cnt, idxf, imskf, gm2, n),
            _floor_design_gram(stack_cp, comb, mask2n, n, armrep, cnt, idxf, imskf, gm2, n),
        )
        for t in range(3):
            gy = gy_by_t[t]
            gd, k_cross = design_by_var[0 if t == 0 else 1]
            gs = torch.where(
                gm2, _gather_sub(gy[armrep], idxf, idxf), torch.zeros((), dtype=dt, device=dev)
            )
            c_f = _gram_pca_coeff(gs, imskf, m_real, cap, k_ref)  # (B, k_ref, mF)
            y64f = _right_center(gs, wf, imskf) @ c_f.transpose(1, 2)  # (B, mF, k_ref)
            (pred64,) = _batched_press_ridge(gd, y64f, [k_cross], lambdas)  # (B, n, k_ref)
            if r_hat is not None:
                g = torch.gather(yr_by_t[t][armrep], 1, idxf) * imskf.to(dt)
                gmean = g.sum(-1) / m_real.clamp(min=1.0)
                ycr = (g - gmean.unsqueeze(-1)) * imskf.to(dt)
                qf = torch.einsum("bkm,bm->bk", c_f, ycr)
                predr = torch.einsum("bgk,bk->bg", pred64, qf).view(n_arms, n_p, 2, n)
                d = (predr[:, :, 0] - predr[:, :, 1]).abs()  # (A, P, n)
            else:
                pred64v = pred64.view(n_arms, n_p, 2, n, k_ref)
                pa, pb = pred64v[:, :, 0], pred64v[:, :, 1]
                t1 = (pa * pa).sum(-1)
                t2 = (pb * pb).sum(-1)
                c_v = c_f.view(n_arms, n_p, 2, k_ref, m_fit_max)
                ia = idx[:, :, 0].reshape(n_arms * n_p, m_fit_max)
                ib = idx[:, :, 1].reshape(n_arms * n_p, m_fit_max)
                ma_ = imsk[:, :, 0]
                mb_ = imsk[:, :, 1]
                gy_e = gy.unsqueeze(1).expand(-1, n_p, -1, -1).reshape(n_arms * n_p, n, n)
                gab = _gather_sub(gy_e, ia, ib).view(n_arms, n_p, m_fit_max, m_fit_max)
                wa = ma_.to(dt) / ma_.sum(-1, keepdim=True).clamp(min=1)
                wb = mb_.to(dt) / mb_.sum(-1, keepdim=True).clamp(min=1)
                cm = torch.einsum("apij,api->apj", gab, wa)
                rm2 = torch.einsum("apij,apj->api", gab, wb)
                gmn = torch.einsum("api,api->ap", rm2, wa)
                gc_ab = gab - cm.unsqueeze(-2) - rm2.unsqueeze(-1) + gmn.unsqueeze(-1).unsqueeze(-1)
                gc_ab = gc_ab * ma_.to(dt).unsqueeze(-1) * mb_.to(dt).unsqueeze(-2)
                m_ab = torch.einsum("apki,apij,aplj->apkl", c_v[:, :, 0], gc_ab, c_v[:, :, 1])
                t12 = torch.einsum("apgk,apkl,apgl->apg", pa, m_ab, pb)
                d = (t1 + t2 - 2.0 * t12).clamp(min=0).sqrt()
            stats[:, t, pc] = _masked_median(d, mask.unsqueeze(1).expand(-1, n_p, -1))

    if r_hat is not None:
        # em/fact/syco: SD-combined floor (floor_sd == np.std ddof=1; <2 pairs → 0.0).
        if n_refit_pairs >= 2:
            fl = stats.std(dim=-1, correction=1)
        else:
            fl = torch.zeros((n_arms, 3), dtype=dt, device=dev)
        floor = fl.max(dim=1).values
    else:
        # marker read-1: p95-combined floor (np.percentile linear == torch.quantile).
        floor = torch.quantile(stats, 0.95, dim=-1).max(dim=1).values
    dof = torch.where(floor >= 1e-12, delta_med / floor, torch.full_like(delta_med, float("nan")))
    dm = delta_med.cpu().numpy()
    dofn = dof.cpu().numpy()
    return [(float(dm[i]), float(dofn[i])) for i in range(n_arms)]


def substrate_swap_null(
    behavior: str,
    substrate: str,
    reduced_root: Path,
    r_hat: np.ndarray | None,
    n_resamples: int,
    *,
    n_refit_pairs: int = NULL_REFIT_PAIRS,
    serial: bool = False,
    device: str = "cpu",
    arm_chunk: int = 16,
    pair_chunk: int = 4,
) -> dict:
    """Matched-n substrate-swap null in the REGISTERED Δ/floor space (B2).

    Reads ``per_question_L{HEADLINE}.npz`` (flat headline-layer rows + per-row context
    index + per-context family). Per resample: draw the substrate's question indices
    with replacement, split them into two matched-n pseudo-substrate halves,
    question-average each half per context → a pseudo-map pair, compute BOTH the raw
    Δ_med AND the normalized Δ/floor (each pseudo-arm refits its own floor through the
    shared harness) for each half, and record ``|Δ(A) − Δ(B)|`` in BOTH spaces. The
    95th percentile of the Δ/floor diffs is X_reg — the REGISTERED threshold a real
    substrate difference in Δ/floor must clear (plan §3/§6/§6.5); the raw-Δ percentiles
    are kept for continuity/diagnostics. The full per-resample Δ/floor null array is
    persisted (``null_delta_over_floor_diffs``) so the analyzer can reconstruct the
    registered band post-hoc.

    Matched-n: both pseudo-arms use the SAME per-half question count, so em's small
    pool yields a WIDE (conservative) null, never an artificially tight one.

    The DEFAULT implementation is the batched Gram-space engine (see the engine
    section above); ``serial=True`` runs the tombstoned per-fit reference (the
    ``--serial-null`` verification escape hatch). Both draw the identical rng
    sequences; ``null_impl`` records which produced the artifact.
    """
    pq_path = reduced_root / behavior / substrate / f"per_question_L{HEADLINE_LAYER}.npz"
    if not pq_path.exists():
        raise FileNotFoundError(f"per-question headline rows missing: {pq_path}")
    d = np.load(pq_path, allow_pickle=True)
    # Fail loud before any keyed read (bug-class-sweep sibling of the round-2
    # `perlayer-npz-key-coverage-preflight` closure; keys = exactly what this site reads).
    _require_npz_keys(
        pq_path,
        d,
        (
            "c_C_base",
            "c_C_trained",
            "v_A_base",
            "v_A_trained",
            "row_context_index",
            "row_question_index",
            "families",
        ),
    )
    c0 = np.asarray(d["c_C_base"], dtype=np.float64)  # (n_rows, HIDDEN)
    cp = np.asarray(d["c_C_trained"], dtype=np.float64)
    v0 = np.asarray(d["v_A_base"], dtype=np.float64)
    vp = np.asarray(d["v_A_trained"], dtype=np.float64)
    row_ctx = np.asarray(d["row_context_index"], dtype=np.int64)
    row_q = np.asarray(d["row_question_index"], dtype=np.int64)
    # families is full-length, indexed by ORIGINAL context index (savemaps writes it so).
    ctx_families = [str(x) for x in d["families"]]
    q_ids = sorted(set(row_q.tolist()))
    n_q = len(q_ids)
    empty = {
        "null_p95": None,
        "null_p975": None,
        "null_over_floor_p95": None,
        "null_over_floor_p975": None,
        "n_questions": n_q,
        "n_resamples_used": 0,
        "null_space": "delta_over_floor",
        # Regime stamp so every future artifact self-describes (resume-predicate keying).
        "n_refit_pairs": n_refit_pairs,
        "n_null_resamples_requested": n_resamples,
    }
    if n_q < 4:
        return {**empty, "note": "too few questions (<4) for a matched-n split"}

    impl = "serial" if serial else "batched_gram_v1"
    if serial:
        _serial_battery_tombstone("substrate_swap_null")
        diffs, dof_diffs = _substrate_swap_null_serial(
            c0,
            cp,
            v0,
            vp,
            row_ctx,
            row_q,
            ctx_families,
            q_ids,
            n_q,
            n_resamples,
            r_hat,
            n_refit_pairs=n_refit_pairs,
        )
    else:
        diffs, dof_diffs = _substrate_swap_null_batched(
            c0,
            cp,
            v0,
            vp,
            row_ctx,
            row_q,
            ctx_families,
            q_ids,
            n_q,
            n_resamples,
            r_hat,
            n_refit_pairs=n_refit_pairs,
            device=device,
            arm_chunk=arm_chunk,
            pair_chunk=pair_chunk,
        )
    if not dof_diffs:
        return {**empty, "null_impl": impl, "note": "all resamples degenerate or floor-underflowed"}
    raw = np.asarray(diffs, dtype=np.float64)
    dof = np.asarray(dof_diffs, dtype=np.float64)
    return {
        # REGISTERED Δ/floor null (the band the verdict + pairwise diff are judged against)
        "null_space": "delta_over_floor",
        "null_over_floor_p95": float(np.percentile(dof, 95)),
        "null_over_floor_p975": float(np.percentile(dof, 97.5)),
        "null_over_floor_median": float(np.median(dof)),
        # full per-resample Δ/floor null array (post-hoc band reconstruction)
        "null_delta_over_floor_diffs": dof.tolist(),
        "n_over_floor_resamples_used": len(dof_diffs),
        # raw Δ_med null (diagnostic / continuity only — NOT the registered band)
        "null_p95": float(np.percentile(raw, 95)) if raw.size else None,
        "null_p975": float(np.percentile(raw, 97.5)) if raw.size else None,
        "null_median": float(np.median(raw)) if raw.size else None,
        "n_questions": n_q,
        "n_resamples_used": len(diffs),
        "n_refit_pairs": n_refit_pairs,
        # Regime stamp so every future artifact self-describes (resume-predicate keying).
        "n_null_resamples_requested": n_resamples,
        # Which engine produced this artifact (additive; the resume predicate ignores it).
        "null_impl": impl,
    }


def _substrate_swap_null_serial(
    c0: np.ndarray,
    cp: np.ndarray,
    v0: np.ndarray,
    vp: np.ndarray,
    row_ctx: np.ndarray,
    row_q: np.ndarray,
    ctx_families: list[str],
    q_ids: list[int],
    n_q: int,
    n_resamples: int,
    r_hat: np.ndarray | None,
    *,
    n_refit_pairs: int,
) -> tuple[list[float], list[float]]:
    """The SUPERSEDED serial per-fit battery — kept VERBATIM as the tombstoned twin.

    One `_pseudo_delta_over_floor` (≈ 2 observed ridge fits + 3 floors ×
    n_refit_pairs × 2 refits) per pseudo-arm per resample, serially. Reached only
    via ``--serial-null`` (equivalence verification) and the batched engine's
    rare per-chunk linear-algebra fallback; `_serial_battery_tombstone` guards the
    flag path. Returns (raw Δ_med diffs, Δ/floor diffs).
    """
    # Map (context, question) → row index for fast per-half question-averaging.
    rc_index: dict[tuple[int, int], int] = {}
    for i in range(len(row_ctx)):
        rc_index[(int(row_ctx[i]), int(row_q[i]))] = i
    contexts = sorted(set(row_ctx.tolist()))

    def _pseudo_stack(
        q_subset: list[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Question-average the subset per context → (n_ctx_kept, HIDDEN) stacks + families."""
        rows_c0, rows_cp, rows_v0, rows_vp, fams = [], [], [], [], []
        for ctx in contexts:
            idxs = [rc_index[(ctx, q)] for q in q_subset if (ctx, q) in rc_index]
            if not idxs:
                continue
            rows_c0.append(c0[idxs].mean(0))
            rows_cp.append(cp[idxs].mean(0))
            rows_v0.append(v0[idxs].mean(0))
            rows_vp.append(vp[idxs].mean(0))
            fams.append(ctx_families[ctx])  # family of THIS battery context
        return (np.stack(rows_c0), np.stack(rows_cp), np.stack(rows_v0), np.stack(rows_vp), fams)

    rng = np.random.default_rng(NULL_SEED)
    half = n_q // 2
    diffs: list[float] = []  # raw Δ_med diffs (diagnostic, continuity)
    dof_diffs: list[float] = []  # Δ/floor diffs (the REGISTERED null space)
    for _ in range(n_resamples):
        drawn = rng.choice(q_ids, size=n_q, replace=True).tolist()
        # split the RESAMPLED question list into two matched-n halves.
        a_qs = drawn[:half]
        b_qs = drawn[half : 2 * half]
        try:
            sa = _pseudo_stack(a_qs)
            sb = _pseudo_stack(b_qs)
            if sa[0].shape[0] < 4 or sb[0].shape[0] < 4:
                continue  # a degenerate half (too few contexts covered) — skip
            da_med, da_dof = _pseudo_delta_over_floor(
                sa[0], sa[1], sa[2], sa[3], sa[4], r_hat, n_refit_pairs=n_refit_pairs
            )
            db_med, db_dof = _pseudo_delta_over_floor(
                sb[0], sb[1], sb[2], sb[3], sb[4], r_hat, n_refit_pairs=n_refit_pairs
            )
        except np.linalg.LinAlgError:
            continue  # degenerate resample geometry — skip (bootstrap noise)
        diffs.append(abs(da_med - db_med))
        if not (np.isnan(da_dof) or np.isnan(db_dof)):
            dof_diffs.append(abs(da_dof - db_dof))
    return diffs, dof_diffs


def _substrate_swap_null_batched(
    c0: np.ndarray,
    cp: np.ndarray,
    v0: np.ndarray,
    vp: np.ndarray,
    row_ctx: np.ndarray,
    row_q: np.ndarray,
    ctx_families: list[str],
    q_ids: list[int],
    n_q: int,
    n_resamples: int,
    r_hat: np.ndarray | None,
    *,
    n_refit_pairs: int,
    device: str,
    arm_chunk: int,
    pair_chunk: int,
) -> tuple[list[float], list[float]]:
    """Batched twin of `_substrate_swap_null_serial` (same draws, same skip semantics).

    Reproduces the serial question-draw sequence EXACTLY (one ``rng.choice`` per
    resample from ``default_rng(NULL_SEED)``), builds every pseudo-arm's
    question-averaged stacks as batched weighted means over a dense
    (context × question × HIDDEN) pool, and evaluates arms through
    `_batched_arm_dofs` in ``arm_chunk``-sized chunks. A resample is skipped iff
    either half keeps <4 contexts (checked BEFORE any fit, like the serial loop)
    or its arm degenerates in the engine's serial fallback (the LinAlgError-skip
    analog). Returns (raw Δ_med diffs, Δ/floor diffs).
    """
    dev = torch.device(device)
    contexts = sorted(set(row_ctx.tolist()))
    n_ctx = len(contexts)
    ctx_pos = {int(c): i for i, c in enumerate(contexts)}
    q_pos = {int(q): i for i, q in enumerate(q_ids)}
    h = c0.shape[1]
    pool = np.zeros((4, n_ctx, n_q, h), dtype=np.float64)
    present = np.zeros((n_ctx, n_q), dtype=np.float64)
    for r in range(len(row_ctx)):
        ci = ctx_pos[int(row_ctx[r])]
        qi = q_pos[int(row_q[r])]
        pool[0, ci, qi] = c0[r]
        pool[1, ci, qi] = cp[r]
        pool[2, ci, qi] = v0[r]
        pool[3, ci, qi] = vp[r]
        present[ci, qi] = 1.0
    fams_by_pos = [str(ctx_families[c]) for c in contexts]

    # Exact serial draw sequence: one choice call per resample, nothing else consumes rng.
    rng = np.random.default_rng(NULL_SEED)
    half = n_q // 2
    draw_counts = np.zeros((n_resamples, 2, n_q), dtype=np.float64)
    for ri in range(n_resamples):
        drawn = rng.choice(q_ids, size=n_q, replace=True).tolist()
        for q in drawn[:half]:
            draw_counts[ri, 0, q_pos[int(q)]] += 1.0
        for q in drawn[half : 2 * half]:
            draw_counts[ri, 1, q_pos[int(q)]] += 1.0

    pool_t = torch.from_numpy(pool).to(dev)
    present_t = torch.from_numpy(present).to(dev)
    cnt_t = torch.from_numpy(draw_counts.reshape(n_resamples * 2, n_q)).to(dev)
    totals = cnt_t @ present_t.T  # (2R, n_ctx) — per-context drawn-question multiplicity
    kept = totals > 0
    kept_counts = kept.sum(-1)

    # Serial semantics: skip the RESAMPLE when either half keeps <4 contexts.
    results: list[tuple[float, float] | None] = [None] * (n_resamples * 2)
    arm_ids = [
        i
        for i in range(n_resamples * 2)
        if int(kept_counts[i]) >= 4 and int(kept_counts[i ^ 1]) >= 4
    ]
    for start in range(0, len(arm_ids), arm_chunk):
        ids = arm_ids[start : start + arm_chunk]
        arms = []
        for i in ids:
            kmask = kept[i]
            wrow = cnt_t[i].unsqueeze(0) * present_t  # (n_ctx, n_q) multiplicities
            num = torch.einsum("cq,scqh->sch", wrow, pool_t)  # (4, n_ctx, H)
            stack = (num / totals[i].clamp(min=1.0).unsqueeze(-1))[:, kmask]
            st = stack.cpu().numpy()
            fams_a = [fams_by_pos[j] for j in range(n_ctx) if bool(kmask[j])]
            arms.append({"c0": st[0], "cplus": st[1], "v0": st[2], "vplus": st[3], "fams": fams_a})
        out = _batched_arm_dofs(
            arms, r_hat, n_refit_pairs=n_refit_pairs, device=device, pair_chunk=pair_chunk
        )
        for i, res in zip(ids, out, strict=True):
            results[i] = res

    diffs: list[float] = []
    dof_diffs: list[float] = []
    for ri in range(n_resamples):
        ra, rb = results[2 * ri], results[2 * ri + 1]
        if ra is None or rb is None:
            continue  # kept<4 half, or a degenerate arm (serial LinAlgError-skip analog)
        da_med, da_dof = ra
        db_med, db_dof = rb
        diffs.append(abs(da_med - db_med))
        if not (math.isnan(da_dof) or math.isnan(db_dof)):
            dof_diffs.append(abs(da_dof - db_dof))
    return diffs, dof_diffs


def _r_hat_for(
    behavior: str, rb_main: dict, rb_fact: dict | None, wu_marker: np.ndarray | None
) -> np.ndarray | None:
    """The read-out direction at the headline layer (None for marker read-1 = ‖ΔM‖)."""
    if behavior == "marker":
        return None  # marker null uses read-1 (‖ΔM‖), behavior-agnostic
    return fitM._r_hat_for(behavior, HEADLINE_LAYER, rb_main, rb_fact)


def _headline_stacks(
    behavior: str, substrate: str, reduced_root: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    """Per-context c_C/v_A stacks (base + trained) at the FROZEN headline layer + keys.

    Reads the reduced 28-layer summary (via the shared CellRecord loader) and slices
    the headline-layer plane into (n_ctx, HIDDEN) stacks — the same object the observed
    ``fit_cell`` / ``fit_marker_layer`` read consumes. Returns
    ``(c0, cplus, v0, vplus, families, context_ids)`` with ``families`` / ``context_ids``
    parallel to the rows (the clustered-bootstrap resampling unit + the pair-alignment
    key across substrates).
    """
    cells = _cells_from_summary(behavior, substrate, HEADLINE_LAYER, reduced_root)
    c0 = np.stack([c.c0 for c in cells]).astype(np.float64)
    cplus = np.stack([c.cplus for c in cells]).astype(np.float64)
    v0 = np.stack([c.v0 for c in cells]).astype(np.float64)
    vplus = np.stack([c.vplus for c in cells]).astype(np.float64)
    families = [str(c.family) for c in cells]
    context_ids = [str(c.source_cid) for c in cells]
    return c0, cplus, v0, vplus, families, context_ids


def _pairwise_signed_diffs_batched(
    drawn_all: list[list[str] | None],
    stack_for,
    r_hat: np.ndarray | None,
    *,
    n_refit_pairs: int,
    device: str,
    arm_chunk: int,
    pair_chunk: int,
) -> list[float]:
    """Batched twin of the serial pairwise-CI resample loop (same skip semantics).

    ``drawn_all`` holds the pre-drawn context resample per resample (None = the
    <4-distinct skip); ``stack_for(ctx_subset, side)`` assembles one side's
    (c0, cplus, v0, vplus, fams) stacks. Every (resample, side) arm runs through
    `_batched_arm_dofs` in chunks (stacks built lazily per chunk to bound memory);
    a resample contributes iff BOTH sides produced a non-NaN Δ/floor (the serial
    LinAlgError / floor-underflow skip analog). Returns the signed Δ/floor diffs.
    """
    jobs = [(ri, side) for ri, d in enumerate(drawn_all) if d is not None for side in ("a", "b")]
    per_res: dict[int, list[float]] = {}
    for start in range(0, len(jobs), 2 * arm_chunk):
        chunk_jobs = jobs[start : start + 2 * arm_chunk]
        arms = []
        for ri, side in chunk_jobs:
            s = stack_for(drawn_all[ri], side)
            arms.append({"c0": s[0], "cplus": s[1], "v0": s[2], "vplus": s[3], "fams": s[4]})
        out = _batched_arm_dofs(
            arms, r_hat, n_refit_pairs=n_refit_pairs, device=device, pair_chunk=pair_chunk
        )
        for (ri, _side), res in zip(chunk_jobs, out, strict=True):
            per_res.setdefault(ri, []).append(float("nan") if res is None else res[1])
    signed_diffs: list[float] = []
    for ri in sorted(per_res):
        vals = per_res[ri]
        if len(vals) == 2 and not (math.isnan(vals[0]) or math.isnan(vals[1])):
            signed_diffs.append(float(vals[0] - vals[1]))
    return signed_diffs


def pairwise_diff_ci(
    behavior: str,
    sub_a: str,
    sub_b: str,
    reduced_root: Path,
    r_hat: np.ndarray | None,
    *,
    n_resamples: int = N_NULL_RESAMPLES,
    n_refit_pairs: int = NULL_REFIT_PAIRS,
    seed: int = NULL_SEED,
    serial: bool = False,
    device: str = "cpu",
    arm_chunk: int = 16,
    pair_chunk: int = 4,
) -> dict:
    """Family-clustered bootstrap CI on the PAIRED Δ/floor difference for one substrate pair.

    D1 (conjunct restore): plan §3 registers "substrate matters" as a CONJUNCTION —
    the max-vs-min substrate Δ/floor difference must (i) exceed the substrate-swap null's
    p95 AND (ii) have a pairwise-difference CI on the Δ/floor difference that EXCLUDES 0.
    The verdict previously gated on (i) alone; this restores (ii).

    Both substrates fit the map over the SAME shared 50-context battery (plan §4.2), so a
    single family-clustered resample of the battery contexts (the ~7-family cluster unit)
    applies IDENTICALLY to both substrate stacks — the paired difference is on the same
    contexts per draw. Per resample: draw whole battery FAMILIES with replacement, restrict
    BOTH substrates to the resampled contexts, refit each substrate's Δ/floor at the frozen
    headline layer via the SHARED refit harness (``_pseudo_delta_over_floor`` — the exact
    ``fit_cell`` / ``fit_marker_layer`` numerator + per-arm refit floor, in each behavior's
    own DV convention: em/fact/syco SD-combined floor, marker read-1 p95-combined floor),
    and record ``Δ/floor(A) − Δ/floor(B)``. The percentile CI on that signed difference is
    the pairwise CI; it EXCLUDES 0 iff ``ci_lo > 0`` or ``ci_hi < 0``.

    The battery contexts are shared, so both substrates key on the same ``context_ids`` —
    a resampled context maps to the same row in each stack via the per-substrate context
    index. Returns the pair record: point ``abs_diff`` (from the FULL-sample observed reads,
    the caller passes these separately for the verdict) plus ``ci_lo`` / ``ci_hi`` /
    ``ci_excludes_zero`` / ``n_families`` / ``n_resamples_used`` from the bootstrap.
    """
    c0a, cpa, v0a, vpa, fam_a, ctx_a = _headline_stacks(behavior, sub_a, reduced_root)
    c0b, cpb, v0b, vpb, _fam_b, ctx_b = _headline_stacks(behavior, sub_b, reduced_root)
    # Both substrates fit over the shared battery, so index each by its OWN context list
    # and resample on the INTERSECTION (a context present in both) — the paired difference
    # requires both arms to cover the drawn context. Families come from the shared battery
    # so they agree per context; use substrate A's family map (identical by construction —
    # substrate B's `_fam_b` would give the same per-context labels).
    a_ctx_to_row = {cid: i for i, cid in enumerate(ctx_a)}
    b_ctx_to_row = {cid: i for i, cid in enumerate(ctx_b)}
    shared_ctx = [cid for cid in ctx_a if cid in b_ctx_to_row]
    ctx_family = {cid: fam_a[a_ctx_to_row[cid]] for cid in shared_ctx}
    empty = {
        "pair": f"{sub_a}_vs_{sub_b}",
        "dv_space": "delta_over_floor",
        "ci_lo": None,
        "ci_hi": None,
        "ci_excludes_zero": None,
        "n_families": len({ctx_family[c] for c in shared_ctx}),
        "n_resamples_used": 0,
    }
    if len(shared_ctx) < 4:
        return {**empty, "note": "too few shared battery contexts (<4) for a paired CI"}

    uniq_fams = sorted({ctx_family[c] for c in shared_ctx})
    fam_to_ctx: dict[str, list[str]] = {f: [] for f in uniq_fams}
    for cid in shared_ctx:
        fam_to_ctx[ctx_family[cid]].append(cid)
    clustered = len(uniq_fams) >= 2
    rng = np.random.default_rng(seed)
    signed_diffs: list[float] = []

    def _stack_for(ctx_subset: list[str], side: str):
        """Assemble (c0, cplus, v0, vplus, families) for one substrate over ctx_subset."""
        rows = (
            [a_ctx_to_row[c] for c in ctx_subset]
            if side == "a"
            else [b_ctx_to_row[c] for c in ctx_subset]
        )
        fams = [ctx_family[c] for c in ctx_subset]
        if side == "a":
            return c0a[rows], cpa[rows], v0a[rows], vpa[rows], fams
        return c0b[rows], cpb[rows], v0b[rows], vpb[rows], fams

    # Pre-draw the context resamples with the EXACT serial rng consumption (one choice
    # call per resample; the <4-distinct check happens AFTER the draw, so a skipped
    # resample still consumed its draw). Both branches below use the same sequence.
    drawn_all: list[list[str] | None] = []
    for _ in range(n_resamples):
        if clustered:
            chosen_fams = rng.choice(uniq_fams, size=len(uniq_fams), replace=True)
            drawn_ctx = [c for f in chosen_fams for c in fam_to_ctx[str(f)]]
        else:
            drawn_ctx = list(rng.choice(shared_ctx, size=len(shared_ctx), replace=True))
        # Need enough distinct contexts on BOTH arms to fit (the refit needs >=4 rows).
        drawn_all.append(drawn_ctx if len(set(drawn_ctx)) >= 4 else None)

    if serial:
        # SUPERSEDED serial reference (tombstoned twin — the --serial-null escape hatch).
        _serial_battery_tombstone("pairwise_diff_ci")
        for drawn_ctx in drawn_all:
            if drawn_ctx is None:
                continue
            try:
                sa = _stack_for(drawn_ctx, "a")
                sb = _stack_for(drawn_ctx, "b")
                _, da_dof = _pseudo_delta_over_floor(
                    sa[0], sa[1], sa[2], sa[3], sa[4], r_hat, n_refit_pairs=n_refit_pairs
                )
                _, db_dof = _pseudo_delta_over_floor(
                    sb[0], sb[1], sb[2], sb[3], sb[4], r_hat, n_refit_pairs=n_refit_pairs
                )
            except np.linalg.LinAlgError:
                continue  # degenerate resample geometry — skip (bootstrap noise)
            if np.isnan(da_dof) or np.isnan(db_dof):
                continue  # a floor underflowed on this resample — excluded
            signed_diffs.append(float(da_dof - db_dof))
    else:
        # Batched default: every (resample, side) arm through the Gram-space engine.
        signed_diffs = _pairwise_signed_diffs_batched(
            drawn_all,
            _stack_for,
            r_hat,
            n_refit_pairs=n_refit_pairs,
            device=device,
            arm_chunk=arm_chunk,
            pair_chunk=pair_chunk,
        )
    if not signed_diffs:
        return {**empty, "note": "all resamples degenerate or floor-underflowed"}
    arr = np.asarray(signed_diffs, dtype=np.float64)
    ci_lo = float(np.percentile(arr, 2.5))
    ci_hi = float(np.percentile(arr, 97.5))
    return {
        "pair": f"{sub_a}_vs_{sub_b}",
        "dv_space": "delta_over_floor",
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        # the CI EXCLUDES 0 iff the whole interval is on one side of 0
        "ci_excludes_zero": bool(ci_lo > 0.0 or ci_hi < 0.0),
        "ci_median": float(np.median(arr)),
        "n_families": len(uniq_fams),
        "n_resamples_used": len(signed_diffs),
        "n_refit_pairs": n_refit_pairs,
    }


def pairwise_substrate_diff_cis(
    observed_by_sub: dict[str, dict],
    behavior: str,
    reduced_root: Path,
    r_hat: np.ndarray | None,
    *,
    n_resamples: int = N_NULL_RESAMPLES,
    n_refit_pairs: int = NULL_REFIT_PAIRS,
    serial: bool = False,
    device: str = "cpu",
    arm_chunk: int = 16,
    pair_chunk: int = 4,
) -> list[dict]:
    """Pairwise Δ/floor difference + family-clustered bootstrap CI across substrate pairs (D1).

    Reports, for each substrate pair in the REGISTERED Δ/floor space (plan §3/§6/§6.5):
    the full-sample ``abs_diff`` = |Δ/floor(A) − Δ/floor(B)| (the observed point the null
    band gates), the SIGNED-difference clustered-bootstrap CI (``ci_lo`` / ``ci_hi`` from
    ``pairwise_diff_ci``), and ``ci_excludes_zero`` — the second conjunct of the plan §3
    "substrate matters" decision rule (H1 requires the CI to INCLUDE 0; H0 requires it to
    EXCLUDE 0). The raw Δ_med difference is carried alongside as a diagnostic only.

    The CI is a genuine bootstrap (not a delegated placeholder): both substrates fit over
    the SAME shared 50-context battery, so a single family-clustered context resample refits
    both arms per draw and the paired Δ/floor difference is recomputed each resample — the
    identical refit machinery the observed read + the substrate-swap null use.
    """
    out = []
    subs = [s for s in SUBSTRATES if s in observed_by_sub]
    for i in range(len(subs)):
        for j in range(i + 1, len(subs)):
            a, b = subs[i], subs[j]
            da = observed_by_sub[a].get("delta_over_floor")
            db = observed_by_sub[b].get("delta_over_floor")
            da_raw = observed_by_sub[a].get("delta_med")
            db_raw = observed_by_sub[b].get("delta_med")
            rec = {
                "pair": f"{a}_vs_{b}",
                "dv_space": "delta_over_floor",
                "delta_over_floor_a": da,
                "delta_over_floor_b": db,
                "abs_diff": (None if (da is None or db is None) else abs(da - db)),
                # raw Δ_med diagnostic (NOT the registered comparison)
                "delta_med_a": da_raw,
                "delta_med_b": db_raw,
                "abs_diff_delta_med": (
                    None if (da_raw is None or db_raw is None) else abs(da_raw - db_raw)
                ),
                "ci_lo": None,
                "ci_hi": None,
                "ci_excludes_zero": None,
            }
            # Only bootstrap the CI when both observed reads exist (a floor-underflowed
            # substrate has no Δ/floor point, so its pairwise diff is undefined).
            if da is not None and db is not None:
                ci = pairwise_diff_ci(
                    behavior,
                    a,
                    b,
                    reduced_root,
                    r_hat,
                    n_resamples=n_resamples,
                    n_refit_pairs=n_refit_pairs,
                    serial=serial,
                    device=device,
                    arm_chunk=arm_chunk,
                    pair_chunk=pair_chunk,
                )
                rec.update(
                    {
                        "ci_lo": ci.get("ci_lo"),
                        "ci_hi": ci.get("ci_hi"),
                        "ci_median": ci.get("ci_median"),
                        "ci_excludes_zero": ci.get("ci_excludes_zero"),
                        "ci_n_families": ci.get("n_families"),
                        "ci_n_resamples_used": ci.get("n_resamples_used"),
                        "ci_note": ci.get("note"),
                    }
                )
            out.append(rec)
    return out


def decide_substrate_matters(
    dofs: dict[str, float | None],
    null_by_sub: dict[str, dict],
    pairwise: list[dict],
) -> dict:
    """The plan §3 CONJUNCTION verdict — pure function (unit-testable, D1 regression).

    Plan §3 registers the decision rule as a CONJUNCTION, NOT the single null-band gate
    the shipped verdict used:

    - **H1 (substrate-agnostic):** per-behavior Δ/floor is INDISTINGUISHABLE across the
      substrates — the max-vs-min Δ/floor difference is WITHIN the substrate-swap null band
      AND every pairwise-difference CI INCLUDES 0.
    - **H0 (substrate matters):** Δ/floor DIFFERS beyond the noise band — the max-vs-min
      difference EXCEEDS the null's p95 AND at least one pairwise-difference CI EXCLUDES 0.

    So:
      substrate_matters = (max_diff > null_x) AND (some pairwise CI excludes 0)
    with the additional gate that the CI-excluding pair must be one that also DRIVES the
    max_diff (its members are the max-vs-min Δ/floor substrates) — a pairwise CI on a pair
    NOT involved in the max spread cannot, on its own, flip the max-vs-min verdict.

    Returns the verdict dict:
      - ``substrate_matters``: True (both conjuncts fire), False (both fail — max within band
        AND all CIs include 0), or None (AMBIGUOUS — exactly one conjunct fires, or an input
        is missing so a conjunct is undecidable).
    ``dofs`` maps substrate → observed Δ/floor (None where floor-underflowed);
    ``null_by_sub`` maps substrate → its substrate-swap-null dict (``null_over_floor_p95``);
    ``pairwise`` is the ``pairwise_substrate_diff_cis`` output.
    """
    valid = {s: v for s, v in dofs.items() if v is not None}
    if len(valid) < 2:
        return {
            "dv_space": "delta_over_floor",
            "decision_rule": _DECISION_RULE,
            "max_vs_min_delta_over_floor_diff": None,
            "null_x_over_floor_p95": None,
            "null_band_conjunct": None,
            "pairwise_ci_conjunct": None,
            "substrate_matters": None,
            "note": "fewer than 2 substrates with a valid Δ/floor point",
        }
    hi_sub = max(valid, key=lambda s: valid[s])
    lo_sub = min(valid, key=lambda s: valid[s])
    max_diff = valid[hi_sub] - valid[lo_sub]
    null_x = max(
        (null_by_sub[s].get("null_over_floor_p95") or 0.0) for s in valid if null_by_sub.get(s)
    )
    # Conjunct (i): the max-vs-min Δ/floor difference clears the substrate-swap null p95.
    null_band_conjunct = (max_diff > null_x) if null_x else None
    # Conjunct (ii): a pairwise CI that EXCLUDES 0 on a DRIVING pair (the {hi,lo} substrates
    # whose difference IS max_diff). A CI on a non-driving pair does not flip the max verdict.
    driving_pair_keys = {f"{hi_sub}_vs_{lo_sub}", f"{lo_sub}_vs_{hi_sub}"}
    driving_recs = [p for p in pairwise if p.get("pair") in driving_pair_keys]
    excludes = [
        p.get("ci_excludes_zero") for p in driving_recs if p.get("ci_excludes_zero") is not None
    ]
    # None when the driving pair's CI could not be computed; else True iff any excludes 0.
    pairwise_ci_conjunct = None if not excludes else any(excludes)
    # Combine the two conjuncts into the tri-state verdict.
    if null_band_conjunct is None or pairwise_ci_conjunct is None:
        matters: bool | None = None  # a conjunct is undecidable → AMBIGUOUS
    elif null_band_conjunct and pairwise_ci_conjunct:
        matters = True  # both fire → substrate matters (H0)
    elif (not null_band_conjunct) and (not pairwise_ci_conjunct):
        matters = False  # both fail → substrate-agnostic (H1)
    else:
        matters = None  # exactly one conjunct fires → AMBIGUOUS (neither H0 nor H1)
    return {
        "dv_space": "delta_over_floor",
        "decision_rule": _DECISION_RULE,
        "max_vs_min_delta_over_floor_diff": max_diff,
        "max_substrate": hi_sub,
        "min_substrate": lo_sub,
        "null_x_over_floor_p95": null_x,
        "null_band_conjunct": null_band_conjunct,
        "pairwise_ci_conjunct": pairwise_ci_conjunct,
        "driving_pair_ci_excludes_zero": excludes,
        "substrate_matters": matters,
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    fit658.DEVICE = fit658._resolve_device("cpu")  # closed-form ridge — CPU by design
    fit658._assert_ridge_exactness()
    logger.info("[phase=analysis] device=%s; ridge exactness gate PASS", fit658.DEVICE)

    ap = argparse.ArgumentParser(
        description="Issue #813 — DVs (Δ/floor + chain-ρ + substrate-swap null)"
    )
    ap.add_argument("--behaviors", nargs="+", default=list(BEHAVIORS), choices=list(BEHAVIORS))
    ap.add_argument("--substrates", nargs="+", default=list(SUBSTRATES), choices=list(SUBSTRATES))
    ap.add_argument(
        "--reduced-root", type=Path, default=PROJECT_ROOT / "eval_results/issue_813/reduced"
    )
    ap.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_813")
    ap.add_argument("--n-null-resamples", type=int, default=N_NULL_RESAMPLES)
    ap.add_argument(
        "--null-refit-pairs",
        type=int,
        default=NULL_REFIT_PAIRS,
        help="per-pseudo-arm refit-floor pairs inside the Δ/floor null (smoke clamps this)",
    )
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="force full recompute of every cell (default: resume-skip cells whose "
        "delta_floor + substrate_swap_null JSONs are already present and complete)",
    )
    ap.add_argument(
        "--serial-null",
        action="store_true",
        help="run the TOMBSTONED serial per-fit substrate-swap battery + pairwise CIs "
        "(verification escape hatch; the batched Gram-space engine is the default and "
        "~50-100x faster; refuses under EPM_FORBID_SERIAL_FITS=1)",
    )
    ap.add_argument(
        "--null-device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="device for the batched null/pairwise engine (auto → cuda when available)",
    )
    ap.add_argument(
        "--null-arm-chunk",
        type=int,
        default=16,
        help="pseudo-arms per batched engine chunk (memory dial; see the engine comment)",
    )
    ap.add_argument(
        "--null-pair-chunk",
        type=int,
        default=4,
        help="refit-floor pairs per inner fit chunk (memory dial: the peak transient is "
        "~arm_chunk x 2*pair_chunk x 2n_ctx x HIDDEN float64 for the standardized stacks)",
    )
    args = ap.parse_args()

    # r_B artifacts (em/syco from #658 r_b.pt; fact from #667 r_b_fact.pt; marker W_U[※]).
    rb_main = fitM._load_rb_main() if any(b in ("em", "sycophancy") for b in args.behaviors) else {}
    rb_fact = fitM._load_rb_fact() if "fact" in args.behaviors else None
    wu_marker = marker_mc.load_wu_marker_direction() if "marker" in args.behaviors else None

    delta_dir = args.out_dir / "delta_floor"
    chain_dir = args.out_dir / "chain_rho"
    null_dir = args.out_dir / "substrate_swap_null"
    for d in (delta_dir, chain_dir, null_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Batched-vs-serial engine selection for the null battery + pairwise CIs (B3).
    null_impl = "serial" if args.serial_null else "batched_gram_v1"
    null_kw = {
        "serial": args.serial_null,
        "device": _resolve_null_device(args.null_device),
        "arm_chunk": args.null_arm_chunk,
        "pair_chunk": args.null_pair_chunk,
    }
    logger.info(
        "[phase=analysis] null battery impl=%s device=%s arm_chunk=%d pair_chunk=%d",
        null_impl,
        null_kw["device"],
        args.null_arm_chunk,
        args.null_pair_chunk,
    )

    per_behavior: dict[str, dict] = {}
    for behavior in args.behaviors:
        observed_by_sub: dict[str, dict] = {}
        null_by_sub: dict[str, dict] = {}
        r_hat = _r_hat_for(behavior, rb_main, rb_fact, wu_marker)
        for substrate in args.substrates:
            # Per-cell resume-skip+load (boundary-swap enabler): a cell whose observed
            # (delta_floor) AND substrate-swap-null JSONs are already present + complete is
            # LOADED from disk into the SAME in-memory structures the downstream
            # pairwise/verdict/summary assembly consumes — never recomputed (one null cell
            # ≈ 2.5h). Any missing/corrupt file or an in-flight null recomputes the cell.
            # The chain_rho JSON (elicit/mix only) is not re-read downstream — obs already
            # carries obs["chain_rho"] — so a skipped cell leaves its existing chain_rho file
            # untouched. --no-resume forces the full recompute path below.
            if not args.no_resume:
                resumed = _resume_cell(
                    behavior,
                    substrate,
                    delta_dir,
                    null_dir,
                    args.n_null_resamples,
                    args.null_refit_pairs,
                )
                if resumed is not None:
                    obs, null = resumed
                    observed_by_sub[substrate] = obs
                    null_by_sub[substrate] = null
                    logger.info(
                        "[phase=analysis] %s/%s resume-skip (delta_floor + null JSONs present)",
                        behavior,
                        substrate,
                    )
                    continue
                # A present-but-regime-mismatched null recomputes (never silently reused).
                stale = _load_cell_json(null_dir / f"{behavior}__{substrate}.json")
                if (
                    stale is not None
                    and (stale.get("n_over_floor_resamples_used", 0) >= 1 or "note" in stale)
                    and not _null_regime_matches(
                        stale, args.n_null_resamples, args.null_refit_pairs
                    )
                ):
                    logger.info(
                        "[phase=analysis] %s/%s resume REJECTED (regime mismatch: "
                        "loaded n_refit_pairs=%s n_null_resamples_requested=%s "
                        "n_resamples_used=%s vs current n_refit_pairs=%d "
                        "n_null_resamples=%d) — recomputing",
                        behavior,
                        substrate,
                        stale.get("n_refit_pairs"),
                        stale.get("n_null_resamples_requested"),
                        stale.get("n_resamples_used"),
                        args.null_refit_pairs,
                        args.n_null_resamples,
                    )
            logger.info(
                "[phase=analysis] observed read %s/%s L%d", behavior, substrate, HEADLINE_LAYER
            )
            obs = observed_read(behavior, substrate, args.reduced_root, rb_main, rb_fact, wu_marker)
            observed_by_sub[substrate] = obs
            (delta_dir / f"{behavior}__{substrate}.json").write_text(
                json.dumps(obs, indent=2, default=float)
            )
            # chain-ρ (elicit+mix only; generic E≈0 → N/A per plan §3/§6)
            if substrate != "generic" and obs.get("chain_rho") is not None:
                (chain_dir / f"{behavior}__{substrate}.json").write_text(
                    json.dumps(
                        {
                            "behavior": behavior,
                            "substrate": substrate,
                            "chain_rho": obs["chain_rho"],
                        },
                        indent=2,
                        default=float,
                    )
                )
            # substrate-swap null (matched-n) at the frozen headline layer
            logger.info("[phase=analysis] substrate-swap null %s/%s", behavior, substrate)
            null = substrate_swap_null(
                behavior,
                substrate,
                args.reduced_root,
                r_hat,
                args.n_null_resamples,
                n_refit_pairs=args.null_refit_pairs,
                **null_kw,
            )
            null_by_sub[substrate] = null
            (null_dir / f"{behavior}__{substrate}.json").write_text(
                json.dumps(null, indent=2, default=float)
            )

        # Pairwise Δ/floor difference + family-clustered bootstrap CI (D1): both the
        # point diff (null-band gate) AND the signed-difference CI (the second conjunct
        # of the plan §3 decision rule — does the CI exclude 0?).
        pairwise = pairwise_substrate_diff_cis(
            observed_by_sub,
            behavior,
            args.reduced_root,
            r_hat,
            n_resamples=args.n_null_resamples,
            n_refit_pairs=args.null_refit_pairs,
            **null_kw,
        )
        per_behavior[behavior] = {
            "observed": observed_by_sub,
            "substrate_swap_null": null_by_sub,
            "pairwise_substrate_diff": pairwise,
        }
        # Verdict per behavior (D1 conjunction restore): plan §3 registers "substrate
        # matters" as (max-vs-min Δ/floor diff > substrate-swap null p95) AND (a
        # driving-pair pairwise-difference CI EXCLUDES 0). The shipped verdict gated on
        # the null-band conjunct ALONE (BLOCKER i813-pairwise-ci-conjunct-missing); the
        # pure `decide_substrate_matters` reducer now enforces BOTH conjuncts. Raw Δ_med
        # is a diagnostic only (NOT the registered comparison — B2).
        dofs = {s: observed_by_sub[s].get("delta_over_floor") for s in observed_by_sub}
        verdict = decide_substrate_matters(dofs, null_by_sub, pairwise)
        # raw Δ_med diagnostic (continuity only)
        raw_valid = {
            s: observed_by_sub[s].get("delta_med")
            for s in observed_by_sub
            if observed_by_sub[s].get("delta_med") is not None
        }
        verdict["max_vs_min_delta_med_diff"] = (
            (max(raw_valid.values()) - min(raw_valid.values())) if len(raw_valid) >= 2 else None
        )
        per_behavior[behavior]["verdict"] = verdict
        logger.info(
            "[phase=analysis] %s: max-min Δ/floor diff=%s vs null X(p95)=%s | "
            "null_band=%s pairwise_CI_excl0=%s → matters=%s",
            behavior,
            verdict.get("max_vs_min_delta_over_floor_diff"),
            verdict.get("null_x_over_floor_p95"),
            verdict.get("null_band_conjunct"),
            verdict.get("pairwise_ci_conjunct"),
            verdict.get("substrate_matters"),
        )

    summary = {
        "issue": 813,
        "read": "map_change_substrate_dependence_M0_vs_Mplus",
        "headline_layer": HEADLINE_LAYER,
        "target_dim": TARGET_DIM,
        # B2: verdict + null are BOTH in the registered Δ/floor space
        "verdict_dv_space": "delta_over_floor",
        # D1: the verdict is the plan §3 CONJUNCTION (null-band AND pairwise-CI),
        # not the single null-band gate the round-1/2 verdict shipped.
        "verdict_decision_rule": _DECISION_RULE,
        "n_null_resamples": args.n_null_resamples,
        "null_refit_pairs": args.null_refit_pairs,
        # Which engine produced the null/pairwise batteries (B3 vectorization; additive).
        "null_impl": null_impl,
        "null_seed": NULL_SEED,
        "git_commit": _git_sha(),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "per_behavior": per_behavior,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    logger.info("[phase=analysis] wrote %s", args.out_dir / "summary.json")
    # NO [phase=done] here — analysis runs as a SUBPROCESS of issue813_dispatch.sh
    # (phase 4), whose stdout it inherits; the poller reserves [phase=done] for the
    # ONE terminal line the .sh emits AFTER the sentinel write (#545). A premature
    # [phase=done] here would false-signal completion before the sentinel exists.
    logger.info("[phase=analysis] analysis complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
