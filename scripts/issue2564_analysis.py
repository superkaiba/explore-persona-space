"""Issue #2564 PE analysis — minimal-pair delta reads against the frozen maps.

Unit 4 of the pre-split build (plan §6 IN FULL + §6.5 deliverable). VM CPU,
fully vectorized (no per-pair Python loops — pair axes are single tensor ops;
bootstrap draws reduce to (B, n_carriers) multiplicity einsums; derangement
nulls gather from precomputed cosine grids).

Inputs (local ``--in-root`` mirror first, else staged from the HF data repo):

- ``analysis_tensors/va2564/va2564_<cell>.pt``   (U2 stores: fp16 va_span +
  va_tail_incl, index rows with draw + boundary fields, ``empty_rows``)
- ``analysis_tensors/vc2564/vc2564_bank.pt``     (fp32 context-end states)
- ``analysis_tensors/embeddings_qwen3_8b/means_anchors.npz`` (U3)
- ``manifests/bank2564_manifest.json``           (frozen bank: 984 contexts /
  2,778 pairs incl. ``changed_tokens``)
- frozen ridge payloads ``issue779_monitoring/n1m_readout/weights/L19/ridge.pt``
  + ``issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt``
  (applied via ``issue779_ffc_n1m_fits.apply_map`` — NOT refit)
- ``eval_results/issue_2564/manipulation_check.json`` (U3 judge gate)

Outputs: ``minpair_delta.json`` (every §6 read + CI + null + ceiling per
axis × arm + the Artifact metadata contract), ``perpair.jsonl`` (per-pair
rows), and prediction tensors → HF
``issue2564_minpair/analysis_tensors/predictions/*.pt``. NO figures (unit 5).

Arms: ``arm_779ce`` (primary frozen map), ``arm_1738ce`` (secondary),
``arm_iddelta`` (identity baseline — the learned bias cancels EXACTLY in the
delta framing, asserted numerically on a random subset via
``mapping_baselines.identity_bias_predict``).

Seeds (plan §9/§11): carrier-clustered bootstrap B=10,000 seed 2215;
derangement null B=10,000 seed 21620; 20 split-half splits seed 2564.

``--round ffr`` (floor-failed re-elicitation, plan v7 — additive; the parent
path is byte-unchanged at default flags): HF rels nest the
``floor_failed_reelicitation`` round segment under each kind root; outputs are
``eval_results/issue_2564/floor-failed-reelicitation/minpair_delta_ffr.json``
+ ``perpair_ffr.jsonl`` + predictions →
``issue2564_minpair/analysis_tensors/floor_failed_reelicitation/predictions``;
the calibration family's PRIMARY ratio denominator is the parent's FROZEN
per-arm global slope (read from the committed parent ``minpair_delta.json``,
``--parent-delta`` override) with the round-pooled slope as companion; text
third-space reads are emitted ``not_collected`` (no Qwen3 embedding capture
in this round).

``--round k100`` (K=100 draw-append on the two low-reliability axes, plan v8
— additive; parent AND ffr paths byte-unchanged at their flags): parent K=10
draws (ids 0-9) are REUSED at the pinned parent HF revision
(``--parent-revision``) and pooled with 90 fresh draws (ids 10-99) staged
from the round-nested ``k100_low_reliability_axes`` prefix, roster-restricted
to the 168 user_fact + query contexts (474 pairs). Adds: dual-source input
staging (``resolve_input(..., source="parent")``), the K=10 bridge gate
(committed-value reproduction to 1e-6), pooled PRIMARY + new-only COMPANION
reliability with a REGISTERED fallback, provenance checks (a) vc parity /
(b) cross-provenance split-half / (c) answer length, a fire recompute at the
realized denominator (12 carriers x 100 draws = 1,200 checks), the r(K)
subsample curve, K-matched text-space pooling for the query cells, and
outputs ``minpair_delta_k100.json`` / ``perpair_k100.jsonl`` under
``eval_results/issue_2564/k100-low-reliability-axes/`` with predictions →
``issue2564_minpair/analysis_tensors/k100_low_reliability_axes/predictions``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + HF token BEFORE torch import (code-style.md)

import numpy as np  # noqa: E402
import torch  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue779_ffc_n1m_fits as N1M  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.experiments.issue2564 import bank2564 as BK  # noqa: E402
from explore_persona_space.experiments.issue2564 import paths as P2564  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2564_analysis")

ISSUE = 2564
HF_DATA_REPO = os.environ.get("EPM_2564_DATA_WRITE_REPO", "superkaiba1/explore-persona-space-data")
HF_PREFIX_FULL = "issue2564_minpair"
HF_PREFIX_SMOKE = "issue2564_minpair/smoke"
HF_PREFIX_SMOKE_FFR = "issue2564_minpair/smoke_ffr"
# floor-failed re-elicitation round (plan v7): HF rels nest the round segment
# under the kind root (run.py hf_round_prefix layout); results live beside the
# parent's under eval_results/issue_2564/floor-failed-reelicitation/.
FFR_ROUND_SEG = "floor_failed_reelicitation"
FFR_RESULTS_DIRNAME = "floor-failed-reelicitation"
# k100 low-reliability-axes round (plan v8): APPENDS 90 fresh draws (ids
# 10-99) to the parent's committed K=10 draws on the low-reliability cells
# (user_fact + query). Dual-source staging: parent rels are fetched at the
# PINNED parent revision (K100_PARENT_REVISION_DEFAULT); k100 rels live under
# the round-nested prefix. Results land beside the parent's under
# eval_results/issue_2564/k100-low-reliability-axes/.
K100_ROUND_SEG = "k100_low_reliability_axes"
K100_RESULTS_DIRNAME = "k100-low-reliability-axes"
HF_PREFIX_SMOKE_K100 = "issue2564_minpair/smoke_k100"
K100_PARENT_REVISION_DEFAULT = "62b1e8889e1a262501937b0ec6f6022e28b4a7e6"
K100_CELLS = ("user_fact", "query")
K100_AXES = ("user_fact", "query_content", "query_form")
K100_N_CONTEXTS = 168  # 120 user_fact + 48 query (12 E + 24 form + 12 qpara)
K100_N_PAIRS = 474  # 120 swap + 120 famswap + 60 install + 60 ipara + 36 form + 12 qpara + 66 qc
K100_DRAWS_TOTAL = 100
K100_DRAW_OFFSET = 10
K100_R_OF_K = (10, 20, 50, 100)  # measured r(K) subsample curve (plan §3b)
# K=10 bridge-gate targets: committed parent reads the restricted-to-draws-0-9
# pooled loader must reproduce to <=1e-6 absolute (plan v8 §7 gate 3; values
# from eval_results/issue_2564/minpair_delta.json, single-turn arm_779ce).
K100_BRIDGE_TOL = 1e-6
K100_BRIDGE_TARGETS = {
    "user_fact": {
        "mean_cos_headline": 0.17208480650441785,
        "r10_mean": 0.13102069808322983,
    },
    "query_form": {
        "mean_cos_headline": 0.30400866272404437,
        "r10_mean": 0.5994338960687404,
    },
}
RIDGE_779_PATH = "issue779_monitoring/n1m_readout/weights/L19/ridge.pt"
RIDGE_1738_PATH = "issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt"

LAYERS = (14, 19, 26)
PRIMARY_LAYER = 19
BOOT_SEED = 2215
NULL_SEED = 21620
SPLIT_SEED = 2564
B_BOOT_DEFAULT = 10_000
B_NULL_DEFAULT = 10_000
N_SPLITS_DEFAULT = 20
FIRE_THRESHOLDS = (50, 70, 90)  # 70 = headline (plan §6 fire rule)

ARMS = ("arm_779ce", "arm_1738ce", "arm_iddelta")
QUERY_AXES = ("query_content", "query_form")
AXES_ALL: tuple[str, ...] = tuple(BK.INSTRUCTION_AXES) + QUERY_AXES

PRIMARY_CLASS_BY_AXIS = {
    **{a: "swap" for a in BK.INSTRUCTION_AXES},
    "query_content": "query_content",
    "query_form": "query_form",
}
PARA_CLASS_BY_AXIS = {
    **{a: "instruction_paraphrase" for a in BK.INSTRUCTION_AXES},
    "query_content": "query_paraphrase",
    "query_form": "query_paraphrase",
}

ORIENTATION_CONVENTIONS = {
    "install": "E -> value (a = bare-E query context, b = value context)",
    "swap": "value_i -> value_j by plan-listed value-index order (i < j)",
    "famswap": "para(value_i) -> para(value_j), same value-index order",
    "instruction_paraphrase": "value -> its paraphrase",
    "query_content": "carrier_i -> carrier_j by carrier-index order (i < j)",
    "query_form": "form_i -> form_j by form-index order E < imp < stmt",
    "query_paraphrase": "E question -> reworded question",
}

DRAW_TO_PAIR_AGGREGATION = (
    "Delta = difference of the two contexts' K=10-draw mean answer vectors "
    "(tail-inclusive L19 primary; empty-completion draws excluded from the "
    "mean, per-context valid-draw counts recorded per pair)"
)
DYADIC_BOOTSTRAP_CONVENTION = (
    "query_content pairs are carrier DYADS: the bootstrap resamples the 12 "
    "carrier VERTICES with replacement and weights each edge by the product "
    "of its sampled endpoint multiplicities (edges with an unsampled endpoint "
    "get weight 0); single-carrier pairs weight by their carrier multiplicity"
)


# ── small numeric helpers (unit-tested) ────────────────────────────────


def rowwise_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine, float64; zero-norm rows -> NaN (counted by callers)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    den = na * nb
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a * b).sum(-1) / den
    return np.where(den > 0, out, np.nan)


def spearman_brown(r_half: np.ndarray | float) -> np.ndarray | float:
    """Split-half -> full-K-mean reliability step-up: r_K = 2 r / (1 + r).

    Guard: r <= -1 (undefined) -> NaN. Vectorized.
    """
    r = np.asarray(r_half, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(r > -1.0, 2.0 * r / (1.0 + r), np.nan)
    return float(out) if np.isscalar(r_half) or out.ndim == 0 else out


def sb_project(r_k: np.ndarray | float, k_from: int, k_to: int) -> np.ndarray | float:
    """General Spearman-Brown projection r_{k_from} -> r_{k_to} via the
    single-draw reliability r1 = r / (k_from - (k_from-1) r) (plan v8 §3b).

    Used by the k100 round: the new-only COMPANION estimator steps 45/45
    split-half to r90 (``spearman_brown``) then projects r90 -> r100 here;
    the pre-registered expectation table projects the committed r10 the same
    way. NaN-safe, vectorized; a zero/negative denominator -> NaN."""
    r = np.asarray(r_k, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        denom = k_from - (k_from - 1) * r
        r1 = np.where(np.abs(denom) > 0, r / denom, np.nan)
        out = k_to * r1 / (1.0 + (k_to - 1) * r1)
    return float(out) if np.isscalar(r_k) or out.ndim == 0 else out


def suppression_verdict(ceiling_pt: float, ci_lo: float, ci_hi: float) -> bool:
    """Plan §6 convention 1: suppress the ceiling-normalized read where the
    ceiling is nonpositive OR its bootstrap CI includes zero (lo <= 0 <= hi).
    A non-finite ceiling is suppressed too."""
    if not np.isfinite(ceiling_pt) or ceiling_pt <= 0.0:
        return True
    if np.isfinite(ci_lo) and np.isfinite(ci_hi) and ci_lo <= 0.0 <= ci_hi:
        return True
    return False


def through_origin_slope(pred_norm: np.ndarray, obs_norm: np.ndarray) -> float:
    """Through-origin OLS slope of ||pred|| on ||obs||: sum(p*o)/sum(o^2)."""
    p = np.asarray(pred_norm, dtype=np.float64)
    o = np.asarray(obs_norm, dtype=np.float64)
    den = float((o * o).sum())
    return float((p * o).sum() / den) if den > 0 else float("nan")


def ols_intercept_slope(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Plain OLS y ~ 1 + x; returns (intercept, slope). Degenerate x -> slope 0."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    den = float(((x - xm) ** 2).sum())
    if den == 0.0:
        return float(ym), 0.0
    slope = float(((x - xm) * (y - ym)).sum() / den)
    return float(ym - slope * xm), slope


def deranged_perms(n: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """(b, n) permutations with NO fixed point (issue2215_analysis idiom)."""
    assert n >= 2, n
    out = np.empty((b, n), dtype=np.int64)
    filled = 0
    while filled < b:
        batch = max(2 * (b - filled) + 8, 16)
        perms = np.argsort(rng.random((batch, n)), axis=1)
        good = perms[(perms != np.arange(n)).all(axis=1)]
        take = good[: b - filled]
        out[filled : filled + len(take)] = take
        filled += len(take)
    return out


def _pct(a: np.ndarray, q: float) -> float:
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).any():
        return float("nan")
    return float(np.nanpercentile(a, q))


def _ci(draws: np.ndarray) -> list[float]:
    return [_pct(draws, 2.5), _pct(draws, 97.5)]


def carrier_multiplicities(idx_draws: np.ndarray, n_carriers: int) -> np.ndarray:
    """(B, n_carriers) resample multiplicities from (B, n_carriers) index draws."""
    return ((idx_draws[:, :, None] == np.arange(n_carriers)[None, None, :]).sum(axis=1)).astype(
        np.float64
    )


def boot_pair_sums(
    vals: np.ndarray,
    ca: np.ndarray,
    cb: np.ndarray,
    dyad: np.ndarray,
    mult: np.ndarray,
) -> np.ndarray:
    """Per-draw weighted sums over pairs: w = mult[:,ca] (single-carrier) or
    mult[:,ca]*mult[:,cb] (dyads). Reduces to (B, n_car) contractions — no
    (B, n_pairs) weight matrix is ever materialized. NaN vals contribute 0
    with 0 weight ONLY if the caller pre-drops them; here NaNs propagate."""
    vals = np.asarray(vals, dtype=np.float64)
    n_car = mult.shape[1]
    out = np.zeros(mult.shape[0], dtype=np.float64)
    single = ~dyad
    if single.any():
        per_c = np.zeros(n_car, dtype=np.float64)
        np.add.at(per_c, ca[single], vals[single])
        out += mult @ per_c
    if dyad.any():
        m = np.zeros((n_car, n_car), dtype=np.float64)
        np.add.at(m, (ca[dyad], cb[dyad]), vals[dyad])
        out += np.einsum("bi,ij,bj->b", mult, m, mult)
    return out


def boot_weighted_mean(
    vals: np.ndarray, ca: np.ndarray, cb: np.ndarray, dyad: np.ndarray, mult: np.ndarray
) -> np.ndarray:
    """Per-draw carrier-clustered weighted means; finite vals only (NaN rows
    are dropped from numerator AND denominator)."""
    vals = np.asarray(vals, dtype=np.float64)
    ok = np.isfinite(vals)
    num = boot_pair_sums(np.where(ok, vals, 0.0), ca, cb, dyad, mult)
    den = boot_pair_sums(ok.astype(np.float64), ca, cb, dyad, mult)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 0, out, np.nan)


def dyad_pair_weights(mult: np.ndarray, ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """(B, n_pairs) dyadic edge weights = product of endpoint multiplicities.
    Exposed for the unit test pinning the dyadic/vertex bootstrap convention."""
    return mult[:, ca] * mult[:, cb]


# ── identity-cancellation assert (plan §6 mapping-baselines pair) ──────


def identity_cancellation_check(
    vc: np.ndarray,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    rng: np.random.Generator,
    n_check: int = 32,
) -> dict:
    """Numeric fitting-free assert that ``arm_iddelta`` IS the identity+
    learned-bias baseline: (x_A + b) - (x_B + b) == x_A - x_B, with b produced
    by the REAL ``identity_bias_predict`` helper on a synthetic train set."""
    n = len(a_idx)
    sel = rng.choice(n, size=min(n_check, n), replace=False)
    d = vc.shape[1]
    x_train = rng.standard_normal((4, d))
    bias = rng.standard_normal(d) * 3.0
    y_train = x_train + bias
    xa = vc[a_idx[sel]]
    xb = vc[b_idx[sel]]
    pred_a = identity_bias_predict(x_train, y_train, xa)
    pred_b = identity_bias_predict(x_train, y_train, xb)
    err = float(np.abs((pred_a - pred_b) - (xa - xb)).max())
    scale = float(max(np.abs(xa - xb).max(), 1.0))
    tol = 1e-8 * scale
    assert err <= tol, f"identity-bias cancellation violated: err={err} > tol={tol}"
    return {
        "n_pairs_checked": int(len(sel)),
        "max_abs_err": err,
        "tol": tol,
        "statement": "identity_bias_predict(x_A) - identity_bias_predict(x_B) == x_A - x_B",
    }


# ── config ─────────────────────────────────────────────────────────────


@dataclass
class CfgPE:
    in_root: Path | None
    out_dir: Path
    stage_dir: Path
    manip_check: Path
    ridge_779: Path | None
    ridge_1738: Path | None
    smoke: bool
    upload: str
    b_boot: int
    b_null: int
    n_splits: int
    hf_prefix: str
    round: str = "parent"
    parent_delta: Path | None = None  # ffr/k100: parent minpair_delta.json (frozen slopes)
    parent_revision: str = K100_PARENT_REVISION_DEFAULT  # k100: pin for PARENT-source inputs
    seed_base: int = BOOT_SEED

    @property
    def is_ffr(self) -> bool:
        return self.round == "ffr"

    @property
    def is_k100(self) -> bool:
        return self.round == "k100"

    @property
    def pred_dir(self) -> Path:
        return self.out_dir / "predictions"

    @property
    def delta_name(self) -> str:
        if self.is_k100:
            return "minpair_delta_k100.json"
        return "minpair_delta_ffr.json" if self.is_ffr else "minpair_delta.json"

    @property
    def perpair_name(self) -> str:
        if self.is_k100:
            return "perpair_k100.jsonl"
        return "perpair_ffr.jsonl" if self.is_ffr else "perpair.jsonl"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0].replace("%", "%%"))
    ap.add_argument("--in-root", type=Path, default=None, help="local pod out-root mirror")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--stage-dir", type=Path, default=None)
    ap.add_argument("--manip-check", type=Path, default=None)
    ap.add_argument("--ridge-779", type=Path, default=None, help="local ridge payload override")
    ap.add_argument("--ridge-1738", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument(
        "--round",
        choices=("parent", "ffr", "k100"),
        default="parent",
        help="ffr = floor-failed re-elicitation round (plan v7): round-nested HF "
        "rels, ffr out names, frozen parent global-slope denominator. "
        "k100 = K=100 draw-append round (plan v8): dual-source staging (parent "
        "draws 0-9 at --parent-revision + fresh draws >= 10 at the "
        "k100_low_reliability_axes round prefix), K=10 bridge gate, pooled + "
        "new-only reliability, r(K) curve, k100 out names",
    )
    ap.add_argument(
        "--parent-delta",
        type=Path,
        default=None,
        help="ffr/k100 only: parent minpair_delta.json carrying the frozen per-arm "
        "global slopes (default: the committed production copy)",
    )
    ap.add_argument(
        "--parent-revision",
        type=str,
        default=K100_PARENT_REVISION_DEFAULT,
        help="k100 only: pinned HF revision for PARENT-source artifacts "
        "(vc/va stores, bank manifest, anchors, per-draw embeddings)",
    )
    ap.add_argument("--upload", choices=("hf", "none"), default=None)
    ap.add_argument("--b-boot", type=int, default=None)
    ap.add_argument("--b-null", type=int, default=None)
    ap.add_argument("--n-splits", type=int, default=N_SPLITS_DEFAULT)
    ap.add_argument("--import-check", action="store_true")
    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_config(args: argparse.Namespace) -> CfgPE:
    """Resolve the CLI namespace (smoke rebinds out-dir/prefix/B, never inputs).

    Default out-dirs come from the shared ``experiments.issue2564.paths``
    constants so the figures consumer's defaults cannot drift (r2 blocker 4).
    Under ``--smoke`` an explicit ``--manip-check`` is REQUIRED (r2 [g5]): the
    production ``manipulation_check.json`` is a different fire regime, and
    silently gating smoke reads on it would mask wiring bugs.
    """
    smoke = bool(args.smoke)
    rnd = str(getattr(args, "round", "parent"))
    is_ffr = rnd == "ffr"
    is_k100 = rnd == "k100"
    if getattr(args, "parent_delta", None) is not None and not (is_ffr or is_k100):
        raise SystemExit("--parent-delta is an ffr/k100-only flag (pass --round ffr|k100)")
    repo_root = P2564.repo_root()
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        base = P2564.smoke_results_dir() if smoke else P2564.production_results_dir()
        if is_ffr:
            out_dir = base / FFR_RESULTS_DIRNAME
        elif is_k100:
            out_dir = base / K100_RESULTS_DIRNAME
        else:
            out_dir = base
    stage_leaf = {
        (False, "parent"): "pe_stage",
        (True, "parent"): "pe_stage_smoke",
        (False, "ffr"): "pe_stage_ffr",
        (True, "ffr"): "pe_stage_ffr_smoke",
        (False, "k100"): "pe_stage_k100",
        (True, "k100"): "pe_stage_k100_smoke",
    }[(smoke, rnd)]
    default_stage = repo_root / "data" / "issue_2564" / "hf_dl" / stage_leaf
    if args.manip_check is not None:
        manip = Path(args.manip_check)
    elif is_k100:
        # k100 (plan v8 K6): the PRIMARY fire gate is RECOMPUTED from the round's
        # anchors at the realized denominator (12 carriers x 100 draws = 1,200
        # checks); the committed PARENT manipulation_check.json is consumed only
        # as the 120-check COMPANION + the K=10 bridge gate's fire source (the
        # exact regime the bridge reproduces), so the r2 [g5] smoke guard below
        # deliberately does not apply — smoke and production both default to it.
        manip = P2564.production_results_dir() / "manipulation_check.json"
    elif smoke:
        raise SystemExit(
            "--manip-check is REQUIRED under --smoke: the smoke run must never silently "
            "read the committed PRODUCTION manipulation_check.json (its fire verdicts come "
            "from a different regime and would gate the smoke reads)."
        )
    elif is_ffr:
        manip = P2564.production_results_dir() / FFR_RESULTS_DIRNAME / "manipulation_check_ffr.json"
    else:
        manip = P2564.production_results_dir() / "manipulation_check.json"
    parent_delta: Path | None = None
    if is_ffr or is_k100:
        parent_delta = (
            Path(args.parent_delta)
            if args.parent_delta is not None
            else P2564.production_results_dir() / "minpair_delta.json"
        )
    if smoke:
        if is_ffr:
            hf_prefix = HF_PREFIX_SMOKE_FFR
        elif is_k100:
            hf_prefix = HF_PREFIX_SMOKE_K100
        else:
            hf_prefix = HF_PREFIX_SMOKE
    else:
        hf_prefix = HF_PREFIX_FULL
    return CfgPE(
        in_root=Path(args.in_root) if args.in_root else None,
        out_dir=out_dir,
        stage_dir=Path(args.stage_dir) if args.stage_dir else default_stage,
        manip_check=manip,
        ridge_779=Path(args.ridge_779) if args.ridge_779 else None,
        ridge_1738=Path(args.ridge_1738) if args.ridge_1738 else None,
        smoke=smoke,
        upload=args.upload if args.upload is not None else ("none" if smoke else "hf"),
        b_boot=int(args.b_boot) if args.b_boot is not None else (100 if smoke else B_BOOT_DEFAULT),
        b_null=int(args.b_null) if args.b_null is not None else (100 if smoke else B_NULL_DEFAULT),
        n_splits=int(args.n_splits),
        hf_prefix=hf_prefix,
        round=rnd,
        parent_delta=parent_delta,
        parent_revision=str(getattr(args, "parent_revision", K100_PARENT_REVISION_DEFAULT)),
    )


# ── input resolution (local-first, else HF stage; fail loud) ───────────


def resolve_input(cfg: CfgPE, rel: str, *, source: str = "round") -> Path:
    """``<in_root>/<rel>`` when present, else stage ``<hf_prefix>/<rel>`` from
    the HF data repo (retried, atomic, idempotent via hub.stage_hub_file).

    Under ``--round ffr`` / ``--round k100`` the HUB rel nests every
    kind-rooted rel (``analysis_tensors/...``, ``manifests/...``) under the
    round segment, mirroring run.py's ``hf_round_prefix`` upload layout — but
    the PRODUCER's local out-root is NOT round-nested (run.py isolates each
    round via a SEPARATE out-root, ``/workspace/eps2564ffr`` /
    ``/workspace/eps2564k100``), so an ``--in-root`` pointed at the pod
    out-root is probed at the producer layout FIRST, then at the HF-mirror
    (nested) layout (r1 blocker ffr-analysis-artifact-path-drift).

    ``source="parent"`` (k100 only, plan v8 K6): stages the PARENT run's copy
    of ``rel`` from the PRODUCTION prefix at the pinned parent revision
    (``cfg.parent_revision``) into ``stage_dir/parent_pin/<revision>/<rel>``
    — the staged path CARRIES the revision identity, so bytes staged under a
    different ``--parent-revision`` can never be reused (r1 blocker
    k100-parent-revision-cache-unkeyed) — with deliberately NO ``in_root``
    probe: the k100 pod out-root carries FRESH k100 artifacts at the same
    producer-layout rel (e.g. the 168-context parity vc), which must never
    shadow the parent's committed copy the pooled/bridge reads consume; and
    under ``--smoke`` the parent files STILL come from the production prefix
    at the pin (plan v8 A8), never ``cfg.hf_prefix``.
    Frozen ridge payloads resolve via resolve_ridge and are round-independent."""
    if source == "parent":
        assert cfg.is_k100, "source='parent' staging is a k100-only path"
        assert cfg.parent_revision, "k100 parent staging requires a non-empty parent revision"
        target = cfg.stage_dir / "parent_pin" / cfg.parent_revision / rel
        if target.exists():
            return target
        from explore_persona_space.orchestrate.hub import stage_hub_file

        logger.info(
            "[pe] staging parent %s/%s@%s from %s",
            HF_PREFIX_FULL,
            rel,
            cfg.parent_revision[:12],
            HF_DATA_REPO,
        )
        return Path(
            stage_hub_file(
                HF_DATA_REPO,
                f"{HF_PREFIX_FULL}/{rel}",
                target,
                revision=cfg.parent_revision,
            )
        )
    assert source == "round", f"unknown resolve_input source {source!r}"
    hub_rel = rel
    seg = FFR_ROUND_SEG if cfg.is_ffr else (K100_ROUND_SEG if cfg.is_k100 else None)
    if seg is not None:
        kind, _, rest = rel.partition("/")
        assert rest, f"kind-rooted rel expected, got {rel!r}"
        hub_rel = f"{kind}/{seg}/{rest}"
    if cfg.in_root is not None:
        for cand_rel in dict.fromkeys((rel, hub_rel)):
            cand = cfg.in_root / cand_rel
            if cand.exists():
                return cand
    target = cfg.stage_dir / hub_rel
    if target.exists():
        return target
    from explore_persona_space.orchestrate.hub import stage_hub_file

    logger.info("[pe] staging %s/%s from %s", cfg.hf_prefix, hub_rel, HF_DATA_REPO)
    return Path(stage_hub_file(HF_DATA_REPO, f"{cfg.hf_prefix}/{hub_rel}", target))


def resolve_ridge(cfg: CfgPE, local: Path | None, repo_path: str) -> Path:
    """Frozen ridge payloads live at issue-owned HF prefixes (NOT under the
    2564 prefix); a local override serves tests / pre-staged copies."""
    if local is not None:
        assert local.exists(), f"ridge payload override missing: {local}"
        return local
    target = cfg.stage_dir / "frozen_maps" / repo_path
    if target.exists():
        return target
    from explore_persona_space.orchestrate.hub import stage_hub_file

    logger.info("[pe] staging frozen map %s", repo_path)
    return Path(stage_hub_file(HF_DATA_REPO, repo_path, target))


def load_ridge_payload(path: Path, expect_d: int, arm: str) -> dict:
    """Load + semantically validate a frozen ridge payload (kind/shape contract)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload.get("kind") == "ridge", (arm, payload.get("kind"))
    w = payload["W"]
    assert tuple(w.shape) == (expect_d, expect_d), (arm, tuple(w.shape), expect_d)
    for k in ("xmu", "xsd", "ymu"):
        assert tuple(payload[k].shape)[-1] == expect_d, (arm, k, tuple(payload[k].shape))
    return payload


# ── store loading ──────────────────────────────────────────────────────


@dataclass
class Stores:
    ctx_ids: list[str]
    row_of: dict[str, int]
    cells: list[str]
    carriers: list[str]  # present carriers, CARRIER_IDS order
    va_tail_mean: dict[int, np.ndarray]  # layer -> (n_ctx, d) float64
    va_span_mean: dict[int, np.ndarray]
    tail_draws: np.ndarray  # (n_ctx, k_max, d) float32, PRIMARY layer
    draw_valid: np.ndarray  # (n_ctx, k_max) bool
    n_valid: np.ndarray  # (n_ctx,)
    ans_len_mean: np.ndarray  # (n_ctx,) mean completion tokens over valid draws
    vc: dict[int, np.ndarray]  # layer -> (n_ctx, d) float64
    emb_mean: np.ndarray | None  # (n_ctx, e) float64; None under ffr (not collected)
    d: int
    input_files: dict = field(default_factory=dict)
    # k100 (plan v8) — new-only (draws >= K100_DRAW_OFFSET) accumulators + the
    # roster's rows in the PARENT vc order (bridge split-score reproduction).
    # All default-None/0 so parent + ffr constructors stay byte-unchanged.
    va_tail_mean_new: dict[int, np.ndarray] | None = None
    va_span_mean_new: dict[int, np.ndarray] | None = None
    ans_len_mean_new: np.ndarray | None = None
    n_valid_new: np.ndarray | None = None
    # new-only text-embedding means (query rows new-only pooled; user_fact rows
    # keep the parent K=10 means — the embed leg is query-only, plan v8 K4).
    emb_mean_new: np.ndarray | None = None
    parent_rows: np.ndarray | None = None  # (n_ctx,) int64 rows into the parent 984 grid
    n_parent_ctx_total: int = 0


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bank_manifest(path: Path, *, is_ffr: bool = False) -> dict:
    """Load + validate the bank manifest. The parent bank pins the frozen
    grid constants; the ffr bank's size is selection-dependent (surviving
    axes x realized widths), so it asserts INTERNAL consistency instead."""
    bank = json.loads(Path(path).read_text())
    if is_ffr:
        assert bank["n_contexts"] == len(bank["contexts"]), (
            bank["n_contexts"],
            len(bank["contexts"]),
        )
        assert bank["n_pairs"] == len(bank["pairs"]), (bank["n_pairs"], len(bank["pairs"]))
    else:
        assert bank["n_contexts"] == BK.N_CONTEXTS and bank["n_pairs"] == BK.N_PAIRS, (
            bank["n_contexts"],
            bank["n_pairs"],
        )
        assert len(bank["pairs"]) == BK.N_PAIRS
    for p in bank["pairs"]:
        assert "changed_tokens" in p, (p["pair_id"], "bank manifest missing changed_tokens")
    return bank


def load_stores(cfg: CfgPE, bank: dict) -> Stores:
    """Assemble per-context mean matrices + per-draw tail tensor (L19) from the
    U2/U3 stores. Empty-completion rows (ZERO vectors in the fp16 store) are
    excluded from every mean; a context with zero valid draws fails loud."""
    files: dict[str, dict] = {}
    vc_path = resolve_input(cfg, "analysis_tensors/vc2564/vc2564_bank.pt")
    vc_store = torch.load(vc_path, map_location="cpu", weights_only=False)
    layers = [int(x) for x in vc_store["layers"]]
    assert tuple(layers) == LAYERS, layers
    ctx_ids = list(vc_store["context_ids"])
    row_of = {cid: i for i, cid in enumerate(ctx_ids)}
    vc_t = vc_store["vc"].to(torch.float64).numpy()
    d = vc_t.shape[2]
    files["vc2564_bank.pt"] = {"path": str(vc_path), "bytes": vc_path.stat().st_size}

    contexts = bank["contexts"]
    missing_bank = [cid for cid in ctx_ids if cid not in contexts]
    assert not missing_bank, f"vc contexts absent from bank manifest: {missing_bank[:5]}"
    cells = sorted({contexts[cid]["cell"] for cid in ctx_ids})
    carriers = [c for c in BK.CARRIER_IDS if c in {contexts[cid]["carrier"] for cid in ctx_ids}]

    n_ctx = len(ctx_ids)
    li = {layer: k for k, layer in enumerate(layers)}
    vc = {layer: np.ascontiguousarray(vc_t[:, li[layer], :]) for layer in LAYERS}

    # per-cell va stores — vectorized accumulation (no per-row Python loop)
    tail_sum = {layer: np.zeros((n_ctx, d), dtype=np.float64) for layer in LAYERS}
    span_sum = {layer: np.zeros((n_ctx, d), dtype=np.float64) for layer in LAYERS}
    len_sum = np.zeros(n_ctx, dtype=np.float64)
    cnt = np.zeros(n_ctx, dtype=np.int64)
    prim_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (ctx, draw, rows_f32)
    k_max = 0
    for cell in cells:
        rel = f"analysis_tensors/va2564/va2564_{cell}.pt"
        p = resolve_input(cfg, rel)
        store = torch.load(p, map_location="cpu", weights_only=False)
        files[f"va2564_{cell}.pt"] = {"path": str(p), "bytes": p.stat().st_size}
        assert [int(x) for x in store["layers"]] == list(LAYERS), store["layers"]
        idx_rows = store["index"]
        tail = store["va_tail_incl"].to(torch.float64).numpy()
        span = store["va_span"].to(torch.float64).numpy()
        assert tail.shape == (len(idx_rows), len(LAYERS), d), (tail.shape, len(idx_rows), d)
        n_rows = len(idx_rows)
        ctx_idx = np.array([row_of.get(rec["context_id"], -1) for rec in idx_rows], dtype=np.int64)
        n_comp = np.array([int(rec["n_completion_tokens"]) for rec in idx_rows], dtype=np.int64)
        draw = np.array([int(rec["draw"]) for rec in idx_rows], dtype=np.int64)
        empty_mask = np.zeros(n_rows, dtype=bool)
        empty_ids = np.array(sorted(int(i) for i in store.get("empty_rows", [])), dtype=np.int64)
        if empty_ids.size:
            empty_mask[empty_ids] = True
        n_absent = int((ctx_idx < 0).sum())
        if n_absent:
            # r2 [g5]: never a silent drop — va rows whose context_id is absent
            # from the vc store are counted loudly (recorded in meta.input_files).
            logger.warning(
                "[pe] va2564_%s: %d/%d rows reference contexts ABSENT from the vc store "
                "(dropped from the join)",
                cell,
                n_absent,
                n_rows,
            )
        files[f"va2564_{cell}.pt"]["n_rows_ctx_absent_from_vc"] = n_absent
        valid = (ctx_idx >= 0) & (n_comp > 0) & ~empty_mask
        for layer in LAYERS:
            np.add.at(tail_sum[layer], ctx_idx[valid], tail[valid, li[layer], :])
            np.add.at(span_sum[layer], ctx_idx[valid], span[valid, li[layer], :])
        np.add.at(len_sum, ctx_idx[valid], n_comp[valid].astype(np.float64))
        np.add.at(cnt, ctx_idx[valid], 1)
        if valid.any():
            k_max = max(k_max, int(draw[valid].max()) + 1)
            prim_chunks.append(
                (ctx_idx[valid], draw[valid], tail[valid, li[PRIMARY_LAYER], :].astype(np.float32))
            )

    zero = [ctx_ids[i] for i in range(n_ctx) if cnt[i] == 0]
    if zero:
        raise RuntimeError(f"contexts with ZERO valid (non-empty) draws: {zero[:10]}")
    va_tail_mean = {layer: tail_sum[layer] / cnt[:, None] for layer in LAYERS}
    va_span_mean = {layer: span_sum[layer] / cnt[:, None] for layer in LAYERS}
    ans_len_mean = len_sum / cnt

    tail_draws = np.zeros((n_ctx, k_max, d), dtype=np.float32)
    draw_valid = np.zeros((n_ctx, k_max), dtype=bool)
    for ctx_v, draw_v, rows_v in prim_chunks:
        key = ctx_v * k_max + draw_v
        assert len(np.unique(key)) == len(key), "duplicate (context, draw) slot within a va store"
        assert not draw_valid[ctx_v, draw_v].any(), "duplicate (context, draw) slot in va stores"
        tail_draws[ctx_v, draw_v] = rows_v
        draw_valid[ctx_v, draw_v] = True

    if cfg.is_ffr:
        # the ffr round collects NO Qwen3-Embedding-8B answer embeddings —
        # text third-space rows are emitted as not_collected (plan v7 §5).
        emb_mean = None
    else:
        emb_path = resolve_input(cfg, "analysis_tensors/embeddings_qwen3_8b/means_anchors.npz")
        with np.load(emb_path, allow_pickle=False) as z:
            emb_ids = [str(x) for x in z["context_ids"].tolist()]
            emb = z["emb_mean"].astype(np.float64)
        files["means_anchors.npz"] = {"path": str(emb_path), "bytes": emb_path.stat().st_size}
        emb_of = {cid: i for i, cid in enumerate(emb_ids)}
        missing_emb = [cid for cid in ctx_ids if cid not in emb_of]
        assert not missing_emb, f"contexts missing from embedding means: {missing_emb[:5]}"
        emb_mean = emb[[emb_of[cid] for cid in ctx_ids]]

    return Stores(
        ctx_ids=ctx_ids,
        row_of=row_of,
        cells=cells,
        carriers=carriers,
        va_tail_mean=va_tail_mean,
        va_span_mean=va_span_mean,
        tail_draws=tail_draws,
        draw_valid=draw_valid,
        n_valid=cnt,
        ans_len_mean=ans_len_mean,
        vc=vc,
        emb_mean=emb_mean,
        d=d,
        input_files=files,
    )


def _dup_keys(keys: list) -> list:
    """Sorted duplicated entries of ``keys`` (r1 blocker
    k100-draw-grid-completeness: duplicate (context_id, draw) keys RAISE in
    every k100 loader — a duplicate silently replacing a missing row keeps
    counts and global draw ranges intact)."""
    seen: set = set()
    dup: set = set()
    for k in keys:
        if k in seen:
            dup.add(k)
        seen.add(k)
    return sorted(dup)


def load_stores_k100(cfg: CfgPE, bank: dict) -> Stores:
    """k100 stores (plan v8 K6): parent committed vc (984 contexts at the
    pinned revision) + DUAL-SOURCE per-cell va stores — parent draws 0-9 at
    the pin, fresh round draws >= 10 at the k100 round prefix — pooled on the
    draw axis and roster-restricted to the contexts the ROUND stores carry
    (168 production / 42 smoke), preserving PARENT vc row order (the bridge
    gate reproduces the parent's split-score rows via ``parent_rows``).

    Asserts (plan v8 §12): A1 — each parent va store's draw-id set is exactly
    {0..K100_DRAW_OFFSET-1} and every roster context is present at ALL parent
    draws; A2 — every round-store draw id is >= K100_DRAW_OFFSET with the
    minimum exactly K100_DRAW_OFFSET. New-only (draws >= offset) accumulators
    land in the ``*_new`` Stores fields for the companion estimator + the
    registered fallback. Query-cell embeddings are K-matched pools of parent
    (draws 0-9) + round (draws >= 10) per-draw rows; user_fact keeps the
    parent K=10 means (plan v8 K4/§11 — the embed leg is query-only)."""
    assert cfg.is_k100
    files: dict[str, dict] = {}
    vc_path = resolve_input(cfg, "analysis_tensors/vc2564/vc2564_bank.pt", source="parent")
    vc_store = torch.load(vc_path, map_location="cpu", weights_only=False)
    layers = [int(x) for x in vc_store["layers"]]
    assert tuple(layers) == LAYERS, layers
    parent_ctx_ids = [str(x) for x in vc_store["context_ids"]]
    parent_row_of = {cid: i for i, cid in enumerate(parent_ctx_ids)}
    vc_t = vc_store["vc"].to(torch.float64).numpy()
    d = vc_t.shape[2]
    files["vc2564_bank.pt"] = {
        "path": str(vc_path),
        "bytes": vc_path.stat().st_size,
        "source": f"parent@{cfg.parent_revision}",
    }

    contexts = bank["contexts"]
    missing_bank = [cid for cid in parent_ctx_ids if cid not in contexts]
    assert not missing_bank, f"vc contexts absent from bank manifest: {missing_bank[:5]}"

    cells = sorted(K100_CELLS)
    stores_by: dict[tuple[str, str], dict] = {}
    for cell in cells:
        rel = f"analysis_tensors/va2564/va2564_{cell}.pt"
        for source in ("parent", "round"):
            p = resolve_input(cfg, rel, source=source)
            store = torch.load(p, map_location="cpu", weights_only=False)
            assert [int(x) for x in store["layers"]] == list(LAYERS), store["layers"]
            key = f"va2564_{cell}.pt" if source == "round" else f"va2564_{cell}.parent.pt"
            files[key] = {"path": str(p), "bytes": p.stat().st_size, "source": source}
            stores_by[(cell, source)] = store

    # roster = contexts the ROUND stores carry (K100 cells only), parent order
    roster_ids: set[str] = set()
    for cell in cells:
        for rec in stores_by[(cell, "round")]["index"]:
            roster_ids.add(str(rec["context_id"]))
    missing_vc = sorted(cid for cid in roster_ids if cid not in parent_row_of)
    assert not missing_vc, f"[k100] round-store contexts absent from parent vc: {missing_vc[:5]}"
    bad_cell = sorted(cid for cid in roster_ids if contexts[cid]["cell"] not in K100_CELLS)
    assert not bad_cell, f"[k100] round-store contexts outside K100 cells: {bad_cell[:5]}"
    parent_rows = np.array(
        [i for i, cid in enumerate(parent_ctx_ids) if cid in roster_ids], dtype=np.int64
    )
    ctx_ids = [parent_ctx_ids[i] for i in parent_rows]
    row_of = {cid: i for i, cid in enumerate(ctx_ids)}
    carriers = [c for c in BK.CARRIER_IDS if c in {contexts[cid]["carrier"] for cid in ctx_ids}]
    n_ctx = len(ctx_ids)
    li = {layer: k for k, layer in enumerate(layers)}
    vc = {layer: np.ascontiguousarray(vc_t[parent_rows][:, li[layer], :]) for layer in LAYERS}

    tail_sum = {layer: np.zeros((n_ctx, d), dtype=np.float64) for layer in LAYERS}
    span_sum = {layer: np.zeros((n_ctx, d), dtype=np.float64) for layer in LAYERS}
    len_sum = np.zeros(n_ctx, dtype=np.float64)
    cnt = np.zeros(n_ctx, dtype=np.int64)
    tail_sum_new = {layer: np.zeros((n_ctx, d), dtype=np.float64) for layer in LAYERS}
    span_sum_new = {layer: np.zeros((n_ctx, d), dtype=np.float64) for layer in LAYERS}
    len_sum_new = np.zeros(n_ctx, dtype=np.float64)
    cnt_new = np.zeros(n_ctx, dtype=np.int64)
    prim_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    k_max = 0
    for cell in cells:
        cell_roster = [cid for cid in ctx_ids if contexts[cid]["cell"] == cell]
        for source in ("parent", "round"):
            store = stores_by[(cell, source)]
            idx_rows = store["index"]
            tail = store["va_tail_incl"].to(torch.float64).numpy()
            span = store["va_span"].to(torch.float64).numpy()
            assert tail.shape == (len(idx_rows), len(LAYERS), d), (tail.shape, len(idx_rows), d)
            n_rows = len(idx_rows)
            ctx_idx = np.array(
                [row_of.get(str(rec["context_id"]), -1) for rec in idx_rows], dtype=np.int64
            )
            n_comp = np.array([int(rec["n_completion_tokens"]) for rec in idx_rows], dtype=np.int64)
            draw = np.array([int(rec["draw"]) for rec in idx_rows], dtype=np.int64)
            empty_mask = np.zeros(n_rows, dtype=bool)
            empty_ids = np.array(
                sorted(int(i) for i in store.get("empty_rows", [])), dtype=np.int64
            )
            if empty_ids.size:
                empty_mask[empty_ids] = True
            draw_set = {int(x) for x in draw.tolist()}
            keys = [(str(rec["context_id"]), int(rec["draw"])) for rec in idx_rows]
            dup = _dup_keys(keys)
            assert not dup, (
                f"[k100] va2564_{cell} ({source}): duplicate (context_id, draw) index "
                f"rows (r1 blocker k100-draw-grid-completeness): {dup[:5]}"
            )
            per_ctx: dict[str, set[int]] = {}
            for cid2, d2 in keys:
                per_ctx.setdefault(cid2, set()).add(d2)
            if source == "parent":
                # A1: parent stores carry EXACTLY draws {0..offset-1}, and every
                # roster context is present at ALL parent draws (index presence;
                # empty-completion rows are still excluded from the means below).
                assert draw_set == set(range(K100_DRAW_OFFSET)), (cell, sorted(draw_set)[:12])
                missing_grain = [
                    cid
                    for cid in cell_roster
                    if per_ctx.get(cid, set()) != set(range(K100_DRAW_OFFSET))
                ]
                assert not missing_grain, (
                    f"[k100] parent va2564_{cell}: roster contexts missing parent draws "
                    f"(A1 full grain): {missing_grain[:5]}"
                )
            else:
                # A2: every fresh draw id sits at/above the offset, min exactly there.
                assert draw_set, f"[k100] round va2564_{cell} carries no rows"
                assert min(draw_set) == K100_DRAW_OFFSET and all(
                    x >= K100_DRAW_OFFSET for x in draw_set
                ), (cell, sorted(draw_set)[:5])
                assert max(draw_set) < K100_DRAWS_TOTAL, (cell, max(draw_set))
                if not cfg.smoke:
                    # production: the fresh grid is EXACTLY draws {offset..99}
                    assert draw_set == set(range(K100_DRAW_OFFSET, K100_DRAWS_TOTAL)), (
                        cell,
                        sorted(set(range(K100_DRAW_OFFSET, K100_DRAWS_TOTAL)) - draw_set)[:5],
                    )
                # A2b (r1 blocker k100-draw-grid-completeness): EVERY roster
                # context carries the store's FULL fresh draw grid (index
                # presence) — a context missing one draw another context
                # carries keeps the GLOBAL set intact and must still raise.
                bad_grain = [cid for cid in cell_roster if per_ctx.get(cid, set()) != draw_set]
                assert not bad_grain, (
                    f"[k100] round va2564_{cell}: contexts missing fresh draws "
                    f"(A2b per-context grid): {bad_grain[:5]}"
                )
            n_absent = int((ctx_idx < 0).sum())
            if n_absent:
                # parent stores legitimately carry non-roster contexts under
                # --smoke (the roster is the round stores' 3-carrier slice);
                # counted loudly either way, never silently dropped.
                logger.warning(
                    "[pe:k100] va2564_%s (%s): %d/%d rows reference contexts outside "
                    "the k100 roster (dropped from the join)",
                    cell,
                    source,
                    n_absent,
                    n_rows,
                )
            key = f"va2564_{cell}.pt" if source == "round" else f"va2564_{cell}.parent.pt"
            files[key]["n_rows_ctx_absent_from_roster"] = n_absent
            valid = (ctx_idx >= 0) & (n_comp > 0) & ~empty_mask
            for layer in LAYERS:
                np.add.at(tail_sum[layer], ctx_idx[valid], tail[valid, li[layer], :])
                np.add.at(span_sum[layer], ctx_idx[valid], span[valid, li[layer], :])
            np.add.at(len_sum, ctx_idx[valid], n_comp[valid].astype(np.float64))
            np.add.at(cnt, ctx_idx[valid], 1)
            if source == "round":
                for layer in LAYERS:
                    np.add.at(tail_sum_new[layer], ctx_idx[valid], tail[valid, li[layer], :])
                    np.add.at(span_sum_new[layer], ctx_idx[valid], span[valid, li[layer], :])
                np.add.at(len_sum_new, ctx_idx[valid], n_comp[valid].astype(np.float64))
                np.add.at(cnt_new, ctx_idx[valid], 1)
            if valid.any():
                k_max = max(k_max, int(draw[valid].max()) + 1)
                prim_chunks.append(
                    (
                        ctx_idx[valid],
                        draw[valid],
                        tail[valid, li[PRIMARY_LAYER], :].astype(np.float32),
                    )
                )

    zero = [ctx_ids[i] for i in range(n_ctx) if cnt[i] == 0]
    if zero:
        raise RuntimeError(f"[k100] contexts with ZERO valid (non-empty) draws: {zero[:10]}")
    zero_new = [ctx_ids[i] for i in range(n_ctx) if cnt_new[i] == 0]
    if zero_new:
        raise RuntimeError(f"[k100] contexts with ZERO valid NEW draws: {zero_new[:10]}")
    assert k_max <= K100_DRAWS_TOTAL, k_max
    if not cfg.smoke:
        assert k_max == K100_DRAWS_TOTAL, f"[k100] production pooled k_max {k_max} != 100"
    va_tail_mean = {layer: tail_sum[layer] / cnt[:, None] for layer in LAYERS}
    va_span_mean = {layer: span_sum[layer] / cnt[:, None] for layer in LAYERS}
    ans_len_mean = len_sum / cnt
    va_tail_mean_new = {layer: tail_sum_new[layer] / cnt_new[:, None] for layer in LAYERS}
    va_span_mean_new = {layer: span_sum_new[layer] / cnt_new[:, None] for layer in LAYERS}
    ans_len_mean_new = len_sum_new / cnt_new

    tail_draws = np.zeros((n_ctx, k_max, d), dtype=np.float32)
    draw_valid = np.zeros((n_ctx, k_max), dtype=bool)
    for ctx_v, draw_v, rows_v in prim_chunks:
        key2 = ctx_v * k_max + draw_v
        assert len(np.unique(key2)) == len(key2), "duplicate (context, draw) slot in a va store"
        assert not draw_valid[ctx_v, draw_v].any(), "duplicate (context, draw) slot in va stores"
        tail_draws[ctx_v, draw_v] = rows_v
        draw_valid[ctx_v, draw_v] = True

    # K-matched text embeddings (plan v8 K4): query contexts pool parent
    # (draws 0-9) + round (draws >= 10) per-draw L2-normalized rows into a
    # plain mean (means_anchors convention: NOT re-normalized); user_fact
    # keeps the parent K=10 means verbatim (the embed leg is query-only).
    emb_rel = "analysis_tensors/embeddings_qwen3_8b/means_anchors.npz"
    emb_path = resolve_input(cfg, emb_rel, source="parent")
    with np.load(emb_path, allow_pickle=False) as z:
        emb_ids = [str(x) for x in z["context_ids"].tolist()]
        emb = z["emb_mean"].astype(np.float64)
    files["means_anchors.parent.npz"] = {
        "path": str(emb_path),
        "bytes": emb_path.stat().st_size,
        "source": "parent",
    }
    emb_of = {cid: i for i, cid in enumerate(emb_ids)}
    missing_emb = [cid for cid in ctx_ids if cid not in emb_of]
    assert not missing_emb, (
        f"[k100] contexts missing from parent embedding means: {missing_emb[:5]}"
    )
    emb_mean = emb[[emb_of[cid] for cid in ctx_ids]].copy()

    pd_rel = "analysis_tensors/embeddings_qwen3_8b/perdraw_anchors.npz"
    pooled_counts: dict[str, dict[str, int]] = {}
    e_dim = emb_mean.shape[1]
    emb_sum = np.zeros((n_ctx, e_dim), dtype=np.float64)
    emb_cnt = np.zeros(n_ctx, dtype=np.int64)
    emb_sum_new = np.zeros((n_ctx, e_dim), dtype=np.float64)
    emb_cnt_new = np.zeros(n_ctx, dtype=np.int64)
    query_rows = np.array(
        [row_of[cid] for cid in ctx_ids if contexts[cid]["cell"] == "query"], dtype=np.int64
    )
    for source in ("parent", "round"):
        p = resolve_input(cfg, pd_rel, source=source)
        with np.load(p, allow_pickle=False) as z:
            pd_ids = [str(x) for x in z["context_ids"].tolist()]
            pd_draws = z["draws"].astype(np.int64)
            pd_emb = z["emb"].astype(np.float64)
        files[f"perdraw_anchors.{source}.npz"] = {
            "path": str(p),
            "bytes": p.stat().st_size,
            "source": source,
        }
        # draw-id disjointness across provenances (plan v8 K4)
        if source == "parent":
            assert pd_draws.size and int(pd_draws.max()) < K100_DRAW_OFFSET, (
                "[k100] parent perdraw rows carry draw ids >= offset"
            )
        else:
            assert pd_draws.size and int(pd_draws.min()) >= K100_DRAW_OFFSET, (
                "[k100] round perdraw rows carry parent draw ids"
            )
        ridx = np.array(
            [
                row_of[cid] if (cid in row_of and contexts[cid]["cell"] == "query") else -1
                for cid in pd_ids
            ],
            dtype=np.int64,
        )
        sel = ridx >= 0
        # r1 blocker k100-draw-grid-completeness: unique (row, draw) keys AND
        # exact equality to the non-empty anchor-row key set (draw_valid over
        # the query roster rows at this source's draw range) — a duplicated or
        # missing per-draw embedding row must never silently shift a K-pooled
        # mean while counts still look plausible.
        keys_emb = [(int(r2), int(d2)) for r2, d2 in zip(ridx[sel], pd_draws[sel])]
        dup_emb = _dup_keys(keys_emb)
        assert not dup_emb, (
            f"[k100] perdraw_anchors ({source}): duplicate (context, draw) embedding "
            f"rows: {dup_emb[:5]}"
        )
        lo2, hi2 = (0, K100_DRAW_OFFSET) if source == "parent" else (K100_DRAW_OFFSET, k_max)
        expected_emb = {
            (int(r2), int(d2))
            for r2 in query_rows.tolist()
            for d2 in range(lo2, hi2)
            if draw_valid[r2, d2]
        }
        missing_keys = sorted(expected_emb - set(keys_emb))
        extra_keys = sorted(set(keys_emb) - expected_emb)
        assert not missing_keys and not extra_keys, (
            f"[k100] perdraw_anchors ({source}) key set != non-empty anchor rows: "
            f"missing {missing_keys[:5]}, extra {extra_keys[:5]}"
        )
        np.add.at(emb_sum, ridx[sel], pd_emb[sel])
        np.add.at(emb_cnt, ridx[sel], 1)
        if source == "round":
            np.add.at(emb_sum_new, ridx[sel], pd_emb[sel])
            np.add.at(emb_cnt_new, ridx[sel], 1)
        pooled_counts[source] = {"n_rows_total": len(pd_ids), "n_rows_query_roster": int(sel.sum())}
    missing_pool = [ctx_ids[i] for i in query_rows if emb_cnt[i] == 0]
    assert not missing_pool, (
        f"[k100] query contexts with ZERO pooled per-draw rows: {missing_pool[:5]}"
    )
    missing_pool_new = [ctx_ids[i] for i in query_rows if emb_cnt_new[i] == 0]
    assert not missing_pool_new, (
        f"[k100] query contexts with ZERO NEW per-draw embedding rows: {missing_pool_new[:5]}"
    )
    emb_mean[query_rows] = emb_sum[query_rows] / emb_cnt[query_rows, None]
    # new-only twin for the registered fallback (user_fact rows stay parent K=10)
    emb_mean_new = emb_mean.copy()
    emb_mean_new[query_rows] = emb_sum_new[query_rows] / emb_cnt_new[query_rows, None]
    files["emb_pooling_k100"] = {
        "query": f"pooled per-draw means (parent draws 0-9 + k100 draws >= {K100_DRAW_OFFSET})",
        "user_fact": "parent K=10 means (embed leg is query-only; plan v8 K4/§11)",
        "counts": pooled_counts,
    }

    return Stores(
        ctx_ids=ctx_ids,
        row_of=row_of,
        cells=cells,
        carriers=carriers,
        va_tail_mean=va_tail_mean,
        va_span_mean=va_span_mean,
        tail_draws=tail_draws,
        draw_valid=draw_valid,
        n_valid=cnt,
        ans_len_mean=ans_len_mean,
        vc=vc,
        emb_mean=emb_mean,
        d=d,
        input_files=files,
        va_tail_mean_new=va_tail_mean_new,
        va_span_mean_new=va_span_mean_new,
        ans_len_mean_new=ans_len_mean_new,
        n_valid_new=cnt_new,
        emb_mean_new=emb_mean_new,
        parent_rows=parent_rows,
        n_parent_ctx_total=len(parent_ctx_ids),
    )


# ── pair table ─────────────────────────────────────────────────────────


@dataclass
class PairArrays:
    ids: list[str]
    cls: list[str]
    axis: list[str]  # cell for instruction classes; pair_class for query classes
    value_a: list[str]
    value_b: list[str]
    carrier_str: list[str]
    a: np.ndarray  # ctx row idx
    b: np.ndarray
    ca: np.ndarray  # carrier idx (present-carrier order)
    cb: np.ndarray  # == ca for single-carrier pairs
    dyad: np.ndarray  # bool
    changed: np.ndarray  # int
    orientation: list[str]
    n: int


def build_pair_arrays(
    bank: dict, st: Stores, smoke: bool, *, is_ffr: bool = False, is_k100: bool = False
) -> PairArrays:
    """Restrict the frozen bank's pairs to contexts present in the stores;
    production (non-smoke) asserts FULL coverage — the frozen 2,778/984 grid
    for the parent, the realized bank's own counts for the ffr round, and the
    168-context / 474-pair K100 roster (parent bank restricted to the k100
    cells' contexts; plan v8 §4) for the k100 round."""
    car_of = {c: i for i, c in enumerate(st.carriers)}
    keep: list[dict] = []
    for p in bank["pairs"]:
        if p["a"] in st.row_of and p["b"] in st.row_of:
            keep.append(p)
    if not keep:
        raise RuntimeError("empty pair selection: no bank pair has both contexts in the stores")
    if not smoke:
        if is_k100:
            exp_ctx, exp_pairs = K100_N_CONTEXTS, K100_N_PAIRS
        elif is_ffr:
            exp_ctx, exp_pairs = bank["n_contexts"], bank["n_pairs"]
        else:
            exp_ctx, exp_pairs = BK.N_CONTEXTS, BK.N_PAIRS
        assert len(st.ctx_ids) == exp_ctx, (len(st.ctx_ids), exp_ctx)
        assert len(keep) == exp_pairs, (len(keep), exp_pairs)

    ids, cls, axis, va_, vb_, cstr, orient = [], [], [], [], [], [], []
    a_i, b_i, ca_i, cb_i, dy, chg = [], [], [], [], [], []
    contexts = bank["contexts"]
    for p in keep:
        pc = p["pair_class"]
        ids.append(p["pair_id"])
        cls.append(pc)
        axis.append(p["cell"] if p["cell"] != "query" else pc)
        va_.append(p["value_a"])
        vb_.append(p["value_b"])
        cstr.append(p["carrier"])
        a_i.append(st.row_of[p["a"]])
        b_i.append(st.row_of[p["b"]])
        if pc == "query_content":
            ca_i.append(car_of[p["carrier_a"]])
            cb_i.append(car_of[p["carrier_b"]])
            dy.append(True)
            orient.append(f"{p['carrier_a']}->{p['carrier_b']}")
        else:
            c = car_of[contexts[p["a"]]["carrier"]]
            ca_i.append(c)
            cb_i.append(c)
            dy.append(False)
            orient.append(f"{p['value_a']}->{p['value_b']}")
        chg.append(int(p["changed_tokens"]))
    return PairArrays(
        ids=ids,
        cls=cls,
        axis=axis,
        value_a=va_,
        value_b=vb_,
        carrier_str=cstr,
        a=np.array(a_i, dtype=np.int64),
        b=np.array(b_i, dtype=np.int64),
        ca=np.array(ca_i, dtype=np.int64),
        cb=np.array(cb_i, dtype=np.int64),
        dyad=np.array(dy, dtype=bool),
        changed=np.array(chg, dtype=np.int64),
        orientation=orient,
        n=len(ids),
    )


# ── fire table (compliance layering) ───────────────────────────────────


def fire_tables_from_doc(doc: dict) -> dict:
    """Fire tables from an IN-MEMORY manipulation-check document (the k100
    fire-recompute path builds its doc at 1,200-draw denominator and never
    round-trips it through a file). Behavior-preserving extraction of
    ``load_fire``'s body — the committed-file path is unchanged."""
    fired: dict[int, dict[tuple[str, str], bool]] = {t: {} for t in FIRE_THRESHOLDS}
    rows = {}
    for r in doc.get("value_rows", []):
        key = (r["axis"], r["value_id"])
        rows[key] = r
        fired[70][key] = r["verdict"] == "fired"
        for t in (50, 90):
            fired[t][key] = r["sensitivity"][str(t)] == "fired"
    axis_rows = {r["axis"]: r for r in doc.get("axis_rows", [])}
    return {"fired": fired, "value_rows": rows, "axis_rows": axis_rows, "meta": doc.get("meta", {})}


def load_fire(manip_path: Path) -> dict:
    """Per-(axis, value_id) fire verdicts at each threshold + per-axis summary.

    Axes absent from the manipulation-check slice get NO entries — pairs on
    those axes are unfiltered (fired mask = all; recorded as fire: null)."""
    return fire_tables_from_doc(json.loads(Path(manip_path).read_text()))


def validate_fire_coverage_ffr(bank: dict, fire: dict) -> None:
    """FFR production fire-gate coverage — fail loud BEFORE any tensor load.

    ``pair_fired_mask``'s no-row default (fired) and the axis-floor
    ``axis_rows.get`` default (floor met) are the PARENT's sanctioned
    unchecked-axis semantics; under ffr an incomplete / wrong-round /
    schema-drifted ``manipulation_check_ffr.json`` must never silently admit
    unchecked values (r1 blocker ffr-fire-gate-fails-open). Required coverage
    is derived from the bank's OWN selection: every selected wording (base +
    paraphrase) on every surviving axis has exactly one value row, and every
    surviving axis has a floor-verdict-bearing axis row. The install-side
    bare anchor ``E`` (and query values generally) is the only value
    legitimately without a fire row — it never enters the required set.
    """
    selected = bank.get("selected")
    if not isinstance(selected, dict) or not selected:
        raise RuntimeError(
            "ffr bank manifest missing a non-empty 'selected' map — not a "
            "production ffr bank (all-axes-fail rounds never reach analysis)"
        )
    if fire["meta"].get("round") != "ffr":
        raise RuntimeError(
            "manipulation check is not an ffr document "
            f"(meta.round={fire['meta'].get('round')!r}) — wrong-round "
            "manipulation_check_ffr.json?"
        )
    required: set[tuple[str, str]] = set()
    for axis, vids in selected.items():
        for vid in vids:
            required.add((axis, vid))
            required.add((axis, f"{vid}p"))
    present = set(fire["value_rows"])
    missing = sorted(required - present)
    extra = sorted(present - required)
    if missing or extra:
        raise RuntimeError(
            "ffr manipulation-check value-row coverage mismatch: "
            f"{len(missing)} missing {missing[:6]}, {len(extra)} extra {extra[:6]} "
            "— incomplete or wrong-round manipulation_check_ffr.json"
        )
    bad_axes = sorted(
        axis for axis in selected if "floor_met" not in fire["axis_rows"].get(axis, {})
    )
    if bad_axes:
        raise RuntimeError(
            "ffr manipulation-check axis rows missing floor verdicts for "
            f"surviving axes: {bad_axes} — incomplete or wrong-round "
            "manipulation_check_ffr.json"
        )


def load_parent_frozen_slopes(path: Path) -> dict[str, float]:
    """Per-arm PARENT global slopes (``global_slope_all2778``) from the
    committed parent minpair_delta.json — the FROZEN primary ratio denominator
    for the ffr round (plan v7 §5: the round's own pooled slope over <=792
    pairs is reported beside it as a companion). The parent doc stores one
    global value per axis block; asserted identical across axes."""
    doc = json.loads(Path(path).read_text())
    out: dict[str, float] = {}
    for arm in ARMS:
        vals = {
            float(ax["calibration"][arm]["global_slope_all2778"]) for ax in doc["axes"].values()
        }
        assert len(vals) == 1, (arm, sorted(vals))
        v = vals.pop()
        assert math.isfinite(v) and v > 0, (arm, v)
        out[arm] = v
    return out


def pair_fired_mask(pa: PairArrays, fire: dict, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """(fired_a, fired_b) bool arrays; a value with NO fire row (query values,
    the bare-E side of install pairs, axes outside a smoke slice) counts FIRED
    (unfiltered — the gate only ever REMOVES checked-and-not-fired values)."""
    fmap = fire["fired"][threshold]

    def _ok(axis: str, vid: str) -> bool:
        return fmap.get((axis, vid), True)

    fa = np.array(
        [
            _ok(ax, va) if cl != "query_content" else True
            for ax, va, cl in zip(pa.axis, pa.value_a, pa.cls)
        ],
        dtype=bool,
    )
    fb = np.array(
        [
            _ok(ax, vb) if cl != "query_content" else True
            for ax, vb, cl in zip(pa.axis, pa.value_b, pa.cls)
        ],
        dtype=bool,
    )
    return fa, fb


# ── reliability (20 seeded 5/5 splits) ─────────────────────────────────


def split_half_stats(st: Stores, pa: PairArrays, n_splits: int, *, scores_fn=None) -> dict:
    """Per-pair split-half direction reliability + noise norm at L19 tail.

    Per split: the valid draws of every context are randomly partitioned into
    two halves (floor/ceil for odd counts); Delta_h = half-mean(A) - half-mean(B);
    r = cos(Delta_h1, Delta_h2); noise = ||Delta_h1 - Delta_h2|| / 2. Contexts
    with < 2 valid draws make their pairs NaN (counted). Loop is over the
    n_splits axis only — all pair math is vectorized.

    ``scores_fn(rng, n_ctx, k_max) -> (n_ctx, k_max) float array`` overrides
    the default per-split random scores. The k100 bridge gate uses it to
    reproduce the PARENT round's realized half-assignments: the parent drew
    scores at the (984, 10) grid, so the bridge draws at that shape from the
    identical seed stream and slices the roster's parent rows (plan v8 §6)."""
    draws = st.tail_draws.astype(np.float32)
    valid = st.draw_valid
    n_ctx, k_max, _ = draws.shape
    nv = valid.sum(axis=1)
    ctx_ok = nv >= 2
    pair_ok = ctx_ok[pa.a] & ctx_ok[pa.b]

    r_acc = np.zeros(pa.n, dtype=np.float64)
    noise_acc = np.zeros(pa.n, dtype=np.float64)
    n_used = 0
    for s in range(n_splits):
        rng = np.random.default_rng([SPLIT_SEED, s])
        if scores_fn is None:
            scores = rng.random((n_ctx, k_max))
        else:
            scores = np.array(scores_fn(rng, n_ctx, k_max), dtype=np.float64)
            assert scores.shape == (n_ctx, k_max), (scores.shape, n_ctx, k_max)
        scores[~valid] = np.inf
        order = np.argsort(scores, axis=1)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(np.arange(k_max), (n_ctx, k_max)).copy(), 1)
        half1 = (ranks < (nv // 2)[:, None]) & valid
        half2 = valid & ~half1
        w1 = half1.astype(np.float32)
        w2 = half2.astype(np.float32)
        c1 = np.maximum(w1.sum(1), 1.0)
        c2 = np.maximum(w2.sum(1), 1.0)
        m1 = np.einsum("ck,ckd->cd", w1, draws) / c1[:, None]
        m2 = np.einsum("ck,ckd->cd", w2, draws) / c2[:, None]
        d1 = (m1[pa.a] - m1[pa.b]).astype(np.float64)
        d2 = (m2[pa.a] - m2[pa.b]).astype(np.float64)
        r_acc += np.where(pair_ok, np.nan_to_num(rowwise_cos(d1, d2), nan=0.0), 0.0)
        noise_acc += np.where(pair_ok, np.linalg.norm(d1 - d2, axis=1) / 2.0, 0.0)
        n_used += 1
    r_half = np.where(pair_ok, r_acc / max(n_used, 1), np.nan)
    noise = np.where(pair_ok, noise_acc / max(n_used, 1), np.nan)
    return {
        "r_half": r_half,
        "r_full": spearman_brown(r_half),
        "noise_norm": noise,
        "n_splits": n_splits,
        "n_pairs_insufficient_draws": int((~pair_ok).sum()),
        "half_sizes": "floor(n_valid/2) vs the rest (5/5 at the full K=10)",
    }


# ── axis views ─────────────────────────────────────────────────────────


@dataclass
class AxisView:
    axis: str
    primary_class: str
    para_class: str
    primary_idx: np.ndarray
    para_idx: np.ndarray
    install_idx: np.ndarray | None
    famswap_idx: np.ndarray | None
    primary_grid: np.ndarray | None  # (n_vp, n_car_present) pair idx; None for dyads
    famswap_grid: np.ndarray | None
    primary_vps: list[str]
    null_scheme: str


def _grid_for(pa: PairArrays, sel: np.ndarray, n_car: int) -> tuple[np.ndarray | None, list[str]]:
    """Complete (n_vp, n_car) grid of pair indices for a single-carrier class
    selection; asserts completeness (every vp present for every present carrier)."""
    if sel.size == 0:
        return None, []
    vps = sorted({f"{pa.value_a[i]}-{pa.value_b[i]}" for i in sel})
    vp_of = {v: k for k, v in enumerate(vps)}
    grid = np.full((len(vps), n_car), -1, dtype=np.int64)
    for i in sel:
        grid[vp_of[f"{pa.value_a[i]}-{pa.value_b[i]}"], pa.ca[i]] = i
    assert (grid >= 0).all(), f"incomplete (vp x carrier) grid for pairs {pa.axis[sel[0]]}"
    return grid, vps


def build_axis_views(pa: PairArrays, n_car: int) -> dict[str, AxisView]:
    idx_by = {}
    for i in range(pa.n):
        idx_by.setdefault((pa.axis[i], pa.cls[i]), []).append(i)
    qpara = np.array(
        sorted(i for i in range(pa.n) if pa.cls[i] == "query_paraphrase"), dtype=np.int64
    )
    views: dict[str, AxisView] = {}
    axes_present = sorted({a for a in pa.axis if a != "query_paraphrase"})
    for axis in axes_present:
        prim_cls = PRIMARY_CLASS_BY_AXIS[axis]
        para_cls = PARA_CLASS_BY_AXIS[axis]
        prim = np.array(sorted(idx_by.get((axis, prim_cls), [])), dtype=np.int64)
        if prim.size == 0:
            continue
        if axis in QUERY_AXES:
            para = qpara
            install = None
            fams = None
            fams_grid = None
            if axis == "query_content":
                prim_grid, vps = None, [f"{pa.carrier_str[i]}" for i in prim]
                scheme = (
                    "class-preserving edge derangement over the C(12,2) carrier "
                    "dyads (carrier preservation undefined for dyadic pairs)"
                )
            else:
                prim_grid, vps = _grid_for(pa, prim, n_car)
                scheme = "carrier- and class-preserving form-pair derangement"
        else:
            para = np.array(sorted(idx_by.get((axis, para_cls), [])), dtype=np.int64)
            install = np.array(sorted(idx_by.get((axis, "install"), [])), dtype=np.int64)
            fams = np.array(sorted(idx_by.get((axis, "famswap"), [])), dtype=np.int64)
            prim_grid, vps = _grid_for(pa, prim, n_car)
            fams_grid, fams_vps = _grid_for(pa, fams, n_car) if fams.size else (None, [])
            if fams_grid is not None and prim_grid is not None:
                # r2 [g5]: famswap rows must align 1:1 with the primary grid's
                # rows — famswap pair k is (para(value_a_k), para(value_b_k)),
                # SAME order, paraphrase ids being f"{vid}p" (bank2564 :321-324).
                # cross_family pairs the two grids BY ROW; a sort-order skew
                # would silently compare mismatched value pairs.
                expected_fams_vps = [
                    f"{pa.value_a[prim_grid[k, 0]]}p-{pa.value_b[prim_grid[k, 0]]}p"
                    for k in range(prim_grid.shape[0])
                ]
                assert fams_vps == expected_fams_vps, (
                    f"famswap/primary vp grid misalignment for axis {axis}: "
                    f"{fams_vps} != {expected_fams_vps}"
                )
            if prim_grid is not None and prim_grid.shape[0] >= 2:
                scheme = "carrier- and class-preserving value-pair derangement"
            else:
                scheme = (
                    "sign_randomization_2value — NAMED orientation/sign-randomization "
                    "null (value-pair derangement undefined at C(2,2)=1 swap pair per "
                    "carrier; never a silent cross-carrier fallback)"
                )
        views[axis] = AxisView(
            axis=axis,
            primary_class=prim_cls,
            para_class=para_cls,
            primary_idx=prim,
            para_idx=para,
            install_idx=install,
            famswap_idx=fams,
            primary_grid=prim_grid,
            famswap_grid=fams_grid,
            primary_vps=vps,
            null_scheme=scheme,
        )
    return views


# ── per-axis reads ─────────────────────────────────────────────────────


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = x / n
    return np.where(n > 0, out, 0.0)


def direction_null_draws(
    view: AxisView,
    delta_obs: np.ndarray,
    delta_pred: np.ndarray,
    cos_sel: np.ndarray,
    b_null: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-draw NULL mean direction cos under the axis's registered scheme."""
    if view.primary_grid is not None and view.primary_grid.shape[0] >= 2:
        grid = view.primary_grid
        n_vp, n_car = grid.shape
        u_obs = _unit(delta_obs[grid])  # (n_vp, n_car, d)
        u_pred = _unit(delta_pred[grid])
        cgrid = np.einsum("icd,jcd->ijc", u_pred, u_obs)  # cos(pred_i, obs_j) per carrier
        perms = deranged_perms(n_vp, b_null * n_car, rng).reshape(b_null, n_car, n_vp)
        car_ix = np.arange(n_car)[None, :, None]
        vp_ix = np.arange(n_vp)[None, None, :]
        return cgrid[perms, vp_ix, car_ix].mean(axis=(1, 2))
    if view.axis == "query_content":
        u_obs = _unit(delta_obs[view.primary_idx])
        u_pred = _unit(delta_pred[view.primary_idx])
        cmat = u_pred @ u_obs.T  # (n_e, n_e)
        n_e = cmat.shape[0]
        if n_e < 2:
            return np.full(b_null, np.nan)
        perms = deranged_perms(n_e, b_null, rng)
        return cmat[perms, np.arange(n_e)[None, :]].mean(axis=1)
    # 2-value axes: NAMED sign-randomization null over the primary pairs.
    vals = cos_sel[np.isfinite(cos_sel)]
    if vals.size == 0:
        return np.full(b_null, np.nan)
    signs = rng.integers(0, 2, size=(b_null, vals.size)) * 2 - 1
    return (signs * vals[None, :]).mean(axis=1)


def _nanmedian_quiet(a: np.ndarray, axis: int) -> np.ndarray:
    """nanmedian with the expected All-NaN-slice warning suppressed (a bootstrap
    draw that resamples < 2 distinct carriers legitimately yields NaN)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian(a, axis=axis)


def carrier_mean_cos_median(
    grid_a: np.ndarray,
    grid_b: np.ndarray | None,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    mult: np.ndarray,
    chunk: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point per-vp cos of carrier means (A vs B) + bootstrap draws of the
    per-axis MEDIAN. grid_b=None reuses grid_a on delta_b (same pair grid,
    different delta matrix); otherwise the two grids pair by vp row."""
    gb = grid_a if grid_b is None else grid_b
    da = delta_a[grid_a]  # (n_vp, n_car, d)
    db = delta_b[gb]
    pt = rowwise_cos(da.mean(axis=1), db.mean(axis=1))
    b_tot = mult.shape[0]
    med = np.empty(b_tot, dtype=np.float64)
    for lo in range(0, b_tot, chunk):
        m = mult[lo : lo + chunk]  # (bc, n_car)
        tot = np.maximum(m.sum(axis=1), 1e-12)
        ma = np.einsum("bc,vcd->bvd", m, da) / tot[:, None, None]
        mb = np.einsum("bc,vcd->bvd", m, db) / tot[:, None, None]
        med[lo : lo + chunk] = _nanmedian_quiet(rowwise_cos(ma, mb), axis=1)
    return pt, np.nanmedian(pt[None, :], axis=1), med


def pc1_identity_cos(delta_a: np.ndarray, delta_b: np.ndarray, grid: np.ndarray) -> float:
    """|cos| between the top principal directions (vp-centered) of the
    carrier-mean observed vs predicted delta sets. Exploratory. NaN at n_vp<2."""
    a = delta_a[grid].mean(axis=1)  # (n_vp, d)
    b = delta_b[grid].mean(axis=1)
    if a.shape[0] < 2:
        return float("nan")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    va = np.linalg.svd(a, full_matrices=False)[2][0]
    vb = np.linalg.svd(b, full_matrices=False)[2][0]
    return float(abs(np.dot(va, vb)))


def offdiag_pairmean_cos(x: np.ndarray) -> float:
    """Mean pairwise cosine over rows (off-diagonal), NaN at <2 rows."""
    if x.shape[0] < 2:
        return float("nan")
    u = _unit(x)
    g = u @ u.T
    n = g.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(g[mask].mean())


def boot_pairmean_cos_median(
    grids: np.ndarray, deltas: np.ndarray, idx_draws: np.ndarray, chunk: int = 1000
) -> np.ndarray:
    """Bootstrap (carrier-index resample) of the per-axis MEDIAN over vps of
    the mean pairwise cross-carrier cosine (duplicate carrier draws excluded
    — the issue2215 idiom). grids: (n_vp, n_car) pair indices."""
    u = _unit(deltas[grids])  # (n_vp, n_car, d)
    g = np.einsum("vcd,ved->vce", u, u)  # (n_vp, n_car, n_car)
    b_tot = idx_draws.shape[0]
    out = np.empty(b_tot, dtype=np.float64)
    for lo in range(0, b_tot, chunk):
        ix = idx_draws[lo : lo + chunk]  # (bc, n_car)
        gd = g[:, ix[:, :, None], ix[:, None, :]]  # (n_vp, bc, n_car, n_car)
        distinct = ix[:, :, None] != ix[:, None, :]
        num = (gd * distinct[None]).sum(axis=(2, 3))
        den = distinct.sum(axis=(1, 2))[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = np.where(den > 0, num / np.maximum(den, 1), np.nan)  # (n_vp, bc)
        out[lo : lo + chunk] = _nanmedian_quiet(vals, axis=0)
    return out


# ── k100 round machinery (plan v8) ─────────────────────────────────────

K100_LATTICE_B_FLOOR = 0.55  # r10-CI lower-edge b100 projection, rounded down (plan v8 §3b)
K100_LATTICE_RATIO = 0.35  # midpoint of the two registered c/b predictions (0.475 vs 0.222)
K100_DISSOC_G = 0.15  # half the parent's measured dissociation gap (0.97 - 0.66)
K100_FALLBACK_CRITERION = (
    "PRE-REGISTERED (plan v8 §4; consistency note 2): the new-only estimator becomes "
    "PRIMARY iff provenance check (b)'s cross-provenance vs within-new 95% CIs do NOT "
    "overlap, OR on ANY k100 axis the pooled r100 and the new-only projected r100 95% "
    "carrier-clustered bootstrap CIs do NOT overlap (all-values primary pairs; "
    "non-overlap = lo_1 > hi_2 or lo_2 > hi_1); evaluated informationally under --smoke "
    "(never switches the estimator at smoke n — gate-calibration demotion)"
)


def k100_new_only_stores(st: Stores) -> Stores:
    """REGISTERED-FALLBACK stores twin (plan v8 §4): every pooled read swaps to
    the new-only (draws >= K100_DRAW_OFFSET) estimator — per-context means,
    valid counts, answer lengths, text-embedding means, and the draw axis
    itself (columns sliced past the offset, so split-half runs on the fresh
    draws only)."""
    assert st.va_tail_mean_new is not None and st.n_valid_new is not None
    return replace(
        st,
        va_tail_mean=st.va_tail_mean_new,
        va_span_mean=st.va_span_mean_new,
        ans_len_mean=st.ans_len_mean_new,
        n_valid=st.n_valid_new,
        emb_mean=st.emb_mean_new,
        tail_draws=st.tail_draws[:, K100_DRAW_OFFSET:],
        draw_valid=st.draw_valid[:, K100_DRAW_OFFSET:],
    )


def _k100_parent_only_tail_mean(st: Stores, layer: int) -> np.ndarray:
    """Parent-draws-only (draws < offset) per-context tail mean at LAYER,
    reconstructed exactly from the pooled and new-only float64 sums
    (mean_p = (n*m - n_new*m_new)/n_parent; cancellation error ~1e-12, far
    inside the 1e-6 bridge tolerance)."""
    n_all = st.n_valid.astype(np.float64)
    n_new = st.n_valid_new.astype(np.float64)
    n_par = n_all - n_new
    assert (n_par > 0).all(), "[k100] context with zero valid parent draws"
    num = st.va_tail_mean[layer] * n_all[:, None] - st.va_tail_mean_new[layer] * n_new[:, None]
    return num / n_par[:, None]


def _k100_slot_delta(st: Stores, pa: PairArrays, lo: int, hi: int) -> np.ndarray:
    """Per-pair delta of masked draw-slot means over ``tail_draws[:, lo:hi]``
    (L19 primary; a context with zero valid draws in the slot yields NaN)."""
    hi = min(hi, st.tail_draws.shape[1])
    if hi <= lo:
        return np.full((pa.n, st.tail_draws.shape[2]), np.nan)
    d = st.tail_draws[:, lo:hi].astype(np.float64)
    w = st.draw_valid[:, lo:hi].astype(np.float64)
    cnt = w.sum(axis=1)
    m = np.einsum("ck,ckd->cd", w, d)
    with np.errstate(invalid="ignore", divide="ignore"):
        m = m / cnt[:, None]
    m[cnt == 0] = np.nan
    return m[pa.a] - m[pa.b]


def _ci_overlap(ci1: list[float], ci2: list[float]) -> bool | None:
    """95% CI overlap predicate (None when either CI is non-finite)."""
    vals = [*ci1, *ci2]
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals):
        return None
    return not (ci1[0] > ci2[1] or ci2[0] > ci1[1])


def k100_deciding_ci(draws: np.ndarray, threshold: float) -> dict:
    """95% CI + threshold-straddle fragility for a DECIDING quantity's
    carrier-clustered PAIRED bootstrap draws (plan v8 §3b: "a verdict whose
    deciding quantity's CI straddles its threshold is narrated as fragile";
    r1 blocker k100-verdict-fragility-ci — the ratio/gap is formed PER DRAW
    under the shared carrier resample, so numerator AND denominator
    uncertainty both enter the CI). NaN draws (zero-denominator or
    empty-selection resamples) drop out of the percentile read; an all-NaN
    draw set yields a NaN CI and fragile=None (not evaluable)."""
    draws = np.asarray(draws, dtype=np.float64)
    ci = _ci(draws)
    finite = all(math.isfinite(v) for v in ci)
    return {
        "ci95": ci,
        "threshold": float(threshold),
        "fragile": bool(ci[0] <= threshold <= ci[1]) if finite else None,
        "n_finite_draws": int(np.isfinite(draws).sum()),
        "scheme": "carrier-clustered paired bootstrap (shared resample)",
    }


def k100_bridge_gate(cfg: CfgPE, bank: dict, st: Stores, pa: PairArrays, fire_parent: dict) -> dict:
    """K=10 bridge gate (plan v8 §7 gate 3): the dual-source loader restricted
    to the parent draws 0-9 must reproduce the COMMITTED parent reads.

    Two legs:

    - PER-PAIR parity (binds in BOTH modes — per-pair values are per-context
      quantities, so the 3-carrier smoke slice reproduces them exactly): the
      draws-0-9-restricted split-half r10 and the arm_779ce / arm_iddelta
      direction cosines must match the committed parent ``perpair.jsonl``
      rows to <= K100_BRIDGE_TOL absolute per pair. The parent's realized
      half-assignments are reproduced by drawing the split scores at the
      FULL parent (n_parent_ctx_total, 10) grid and slicing ``parent_rows``.
    - HEADLINE parity (PRODUCTION only; demoted informational under --smoke,
      where a carrier slice cannot reproduce 12-carrier means): user_fact /
      query_form headline mean cos (arm_779ce) + r10 vs K100_BRIDGE_TARGETS.

    A mismatch HALTS the analysis: it means staging or pooling is wrong, not
    that the science changed (pod artifacts persist; the analysis re-runs
    after the fix)."""
    mean_p19 = _k100_parent_only_tail_mean(st, PRIMARY_LAYER)
    obs = mean_p19[pa.a] - mean_p19[pa.b]
    ridge_779 = resolve_ridge(cfg, cfg.ridge_779, RIDGE_779_PATH)
    payload_779 = load_ridge_payload(ridge_779, st.d, "arm_779ce")
    mapped = N1M.apply_map(payload_779, st.vc[PRIMARY_LAYER], torch.device("cpu"))
    cos_by_arm = {
        "arm_779ce": rowwise_cos(mapped[pa.a] - mapped[pa.b], obs),
        "arm_iddelta": rowwise_cos(st.vc[PRIMARY_LAYER][pa.a] - st.vc[PRIMARY_LAYER][pa.b], obs),
    }
    bridge_st = replace(
        st,
        tail_draws=st.tail_draws[:, :K100_DRAW_OFFSET],
        draw_valid=st.draw_valid[:, :K100_DRAW_OFFSET],
    )

    def _parent_grid_scores(rng: np.random.Generator, n_ctx: int, k_max: int) -> np.ndarray:
        # the parent run drew rng.random((984, 10)); slice the roster rows so
        # every context keeps its PARENT-realized half assignment (plan v8 K6)
        assert n_ctx == len(st.parent_rows) and k_max == K100_DRAW_OFFSET, (n_ctx, k_max)
        return rng.random((st.n_parent_ctx_total, k_max))[st.parent_rows]

    rel_b = split_half_stats(bridge_st, pa, cfg.n_splits, scores_fn=_parent_grid_scores)
    r10_b = rel_b["r_full"]

    # per-pair parity vs the committed parent perpair.jsonl (both modes)
    assert cfg.parent_delta is not None
    perpair_path = cfg.parent_delta.parent / "perpair.jsonl"
    assert perpair_path.exists(), f"committed parent perpair.jsonl missing: {perpair_path}"
    committed: dict[str, dict] = {}
    for line in perpair_path.open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            committed[row["pair_id"]] = row

    def _nanfloat(v: object) -> float:
        return float("nan") if v is None else float(v)  # committed NaN -> null (sanitize)

    max_abs: dict[str, float] = {}
    for name, recomputed, getter in (
        ("r10", r10_b, lambda row: _nanfloat(row["r10"])),
        ("cos_arm_779ce", cos_by_arm["arm_779ce"], lambda row: _nanfloat(row["cos"]["arm_779ce"])),
        (
            "cos_arm_iddelta",
            cos_by_arm["arm_iddelta"],
            lambda row: _nanfloat(row["cos"]["arm_iddelta"]),
        ),
    ):
        worst = 0.0
        worst_pair = None
        for i, pid in enumerate(pa.ids):
            row = committed.get(pid)
            assert row is not None, f"[k100-bridge] pair {pid!r} absent from parent perpair.jsonl"
            want, got = getter(row), float(recomputed[i])
            if math.isnan(want) and math.isnan(got):
                continue
            diff = abs(want - got)
            if not math.isfinite(diff) or diff > worst:
                worst, worst_pair = (diff if math.isfinite(diff) else float("inf")), pid
        max_abs[name] = worst
        if worst > K100_BRIDGE_TOL:
            raise RuntimeError(
                f"[k100-bridge] PER-PAIR parity FAILED for {name}: max |diff| {worst:.3e} "
                f"> {K100_BRIDGE_TOL} at pair {worst_pair!r} — staging or pooling is "
                "wrong, not the science (plan v8 §7 gate 3)"
            )

    # headline parity vs the committed axis means (production-binding)
    views = build_axis_views(pa, len(st.carriers))
    fa, fb = pair_fired_mask(pa, fire_parent, 70)
    fired = fa & fb
    headline: dict[str, dict] = {}
    headline_ok_all = True
    for axis, targets in K100_BRIDGE_TARGETS.items():
        view = views.get(axis)
        assert view is not None, f"[k100-bridge] axis {axis!r} missing from the roster views"
        prim = view.primary_idx
        ar = fire_parent["axis_rows"].get(axis)
        floor_met = bool(ar["floor_met"]) if ar is not None else True
        hmask = fired[prim]
        head = prim[hmask] if (floor_met and hmask.any()) else np.array([], dtype=np.int64)
        got_cos = float(np.nanmean(cos_by_arm["arm_779ce"][head])) if head.size else float("nan")
        got_r10 = float(np.nanmean(r10_b[head])) if head.size else float("nan")
        row = {
            "n_headline_pairs": int(head.size),
            "measured": {"mean_cos_headline": got_cos, "r10_mean": got_r10},
            "committed": dict(targets),
            "abs_diff": {
                "mean_cos_headline": abs(got_cos - targets["mean_cos_headline"]),
                "r10_mean": abs(got_r10 - targets["r10_mean"]),
            },
        }
        row["ok"] = all(math.isfinite(v) and v <= K100_BRIDGE_TOL for v in row["abs_diff"].values())
        headline_ok_all &= bool(row["ok"])
        headline[axis] = row
    if not cfg.smoke and not headline_ok_all:
        raise RuntimeError(
            f"[k100-bridge] HEADLINE parity FAILED vs committed targets: {headline} — "
            "staging or pooling is wrong, not the science (plan v8 §7 gate 3)"
        )
    if cfg.smoke and not headline_ok_all:
        logger.warning(
            "[k100-bridge] headline parity demoted under --smoke (carrier slice cannot "
            "reproduce 12-carrier means): %s",
            headline,
        )
    report = {
        "tolerance_abs": K100_BRIDGE_TOL,
        "perpair_parity": {
            "source": str(perpair_path),
            "n_pairs_compared": int(pa.n),
            "max_abs_diff": max_abs,
            "ok": True,
        },
        "headline_parity": {
            "rows": headline,
            "ok": headline_ok_all,
            "demoted": bool(cfg.smoke),
            "note": "binds in production; informational under --smoke (slice arithmetic)",
        },
        "verdict": "pass" if (headline_ok_all or cfg.smoke) else "fail",
    }
    logger.info(
        "[k100-bridge] PASS perpair max|diff|=%s headline_ok=%s (smoke=%s)",
        {k: f"{v:.2e}" for k, v in max_abs.items()},
        headline_ok_all,
        cfg.smoke,
    )
    return report


def k100_provenance_checks(cfg: CfgPE, st: Stores, pa: PairArrays) -> tuple[dict, bool | None]:
    """Provenance-homogeneity checks (plan v8 K6): (a) pod-side vc parity
    re-read, (b) cross-provenance split-half exchangeability, (c) per-context
    answer-length distribution old vs new. Returns ``(block, b_pass)`` —
    ``b_pass`` is None when (b) is not evaluable (smoke draw counts)."""
    # (a) — re-read the pod-side k100_vc_parity.json (already asserted in K3).
    # r1 blocker k100-parent-revision-cache-unkeyed: production REQUIRES a
    # well-formed, non-demoted PASS report whose parent_revision matches THIS
    # run's --parent-revision — a malformed / stale / wrong-revision report is
    # rejected, never read as "not a literal fail".
    vc_path = resolve_input(cfg, "manifests/k100_vc_parity.json")
    vc_rep = json.loads(vc_path.read_text())
    if not isinstance(vc_rep, dict) or vc_rep.get("gate") != "k100_vc_parity":
        raise RuntimeError(f"[k100] malformed pod-side vc parity report at {vc_path}: {vc_rep!r}")
    if vc_rep.get("verdict") == "fail" and not vc_rep.get("demoted"):
        raise RuntimeError(f"[k100] pod-side vc parity report is a FAIL: {vc_rep}")
    if not cfg.smoke:
        if vc_rep.get("verdict") != "pass" or vc_rep.get("demoted"):
            raise RuntimeError(
                "[k100] production requires a non-demoted PASS vc parity report; got "
                f"verdict={vc_rep.get('verdict')!r} demoted={vc_rep.get('demoted')!r} "
                f"at {vc_path}"
            )
        if vc_rep.get("parent_revision") != cfg.parent_revision:
            raise RuntimeError(
                "[k100] vc parity report is STALE: parent_revision "
                f"{vc_rep.get('parent_revision')!r} != --parent-revision "
                f"{cfg.parent_revision!r} (re-run phase B provenance check (a))"
            )
    elif vc_rep.get("verdict") == "pass" and vc_rep.get("parent_revision") != cfg.parent_revision:
        raise RuntimeError(
            "[k100] smoke vc parity report parent_revision "
            f"{vc_rep.get('parent_revision')!r} != --parent-revision {cfg.parent_revision!r}"
        )
    check_a = {
        "source": str(vc_path),
        "verdict": vc_rep.get("verdict"),
        "min_cos": vc_rep.get("min_cos"),
        "min_cos_context": vc_rep.get("min_cos_context"),
        "cos_min_bar": vc_rep.get("cos_min_bar"),
        "demoted": vc_rep.get("demoted"),
        "parent_revision": vc_rep.get("parent_revision"),
    }

    # (b) — cross-provenance vs within-new split-half correlation of shifts
    n_car = len(st.carriers)
    rng = np.random.default_rng([BOOT_SEED, 100])
    idx_draws = rng.integers(0, n_car, size=(cfg.b_boot, n_car))
    mult = carrier_multiplicities(idx_draws, n_car)
    # slot bounds derived from the offset (review r1 minor: never hard-coded)
    off = K100_DRAW_OFFSET
    d_parent = _k100_slot_delta(st, pa, 0, off)
    d_a = _k100_slot_delta(st, pa, off, 2 * off)
    d_b = _k100_slot_delta(st, pa, 2 * off, 3 * off)
    d_c = _k100_slot_delta(st, pa, 3 * off, 4 * off)
    cross = rowwise_cos(d_parent, d_a)
    within = rowwise_cos(d_b, d_c)
    both = np.isfinite(cross) & np.isfinite(within)
    if not both.any():
        check_b = {
            "status": "not_evaluable",
            "note": "insufficient new-draw slots (smoke n) — check (b) needs draws "
            "10-19/20-29/30-39 populated; evaluated in production only",
        }
        b_pass: bool | None = None
    else:
        sel = np.flatnonzero(both)
        cr_pt = float(np.nanmean(cross[sel]))
        wi_pt = float(np.nanmean(within[sel]))
        cr_ci = _ci(boot_weighted_mean(cross[sel], pa.ca[sel], pa.cb[sel], pa.dyad[sel], mult))
        wi_ci = _ci(boot_weighted_mean(within[sel], pa.ca[sel], pa.cb[sel], pa.dyad[sel], mult))
        overlap = _ci_overlap(cr_ci, wi_ci)
        b_pass = bool(overlap) if overlap is not None else None
        check_b = {
            "status": "evaluated",
            "cross_provenance_mean_cos": cr_pt,
            "cross_provenance_ci95": cr_ci,
            "within_new_mean_cos": wi_pt,
            "within_new_ci95": wi_ci,
            "slots": {
                "parent": [0, off],
                "new_a": [off, 2 * off],
                "new_b": [2 * off, 3 * off],
                "new_c": [3 * off, 4 * off],
            },
            "n_pairs_used": int(sel.size),
            "n_pairs_excluded_nan": int((~both).sum()),
            "ci_overlap": overlap,
            "exchangeable": b_pass,
            "bootstrap": {"B": cfg.b_boot, "seed": [BOOT_SEED, 100], "scheme": "carrier-clustered"},
        }

    # (c) — answer-length distribution lives in k100_answer_length_check (needs
    # the bank's context->cell map); assembled by the caller into the same block.
    return {"check_a_vc_parity": check_a, "check_b_cross_provenance": check_b}, b_pass


def k100_reliability_block(cfg: CfgPE, st: Stores, pa: PairArrays, b_pass: bool | None) -> dict:
    """Pooled PRIMARY vs new-only COMPANION reliability + the registered
    fallback decision (plan v8 §4). Per-axis means over ALL primary pairs
    (fire-independent: the estimator comparison is about provenance, not
    compliance) with 95% carrier-clustered bootstrap CIs under a SHARED
    resample, so CI non-overlap reads as estimator disagreement."""
    n_car = len(st.carriers)
    rng = np.random.default_rng([BOOT_SEED, 101])
    idx_draws = rng.integers(0, n_car, size=(cfg.b_boot, n_car))
    mult = carrier_multiplicities(idx_draws, n_car)
    views = build_axis_views(pa, n_car)
    rel_pool = split_half_stats(st, pa, cfg.n_splits)
    new_cols = int(st.tail_draws.shape[1] - K100_DRAW_OFFSET)
    new_st = replace(
        st,
        tail_draws=st.tail_draws[:, K100_DRAW_OFFSET:],
        draw_valid=st.draw_valid[:, K100_DRAW_OFFSET:],
    )
    rel_new = split_half_stats(new_st, pa, cfg.n_splits)
    r_pool = rel_pool["r_full"]
    r_new = sb_project(rel_new["r_full"], new_cols, K100_DRAWS_TOTAL)
    per_axis: dict[str, dict] = {}
    nonoverlap_axes: list[str] = []
    for axis, view in sorted(views.items()):
        prim = view.primary_idx
        pool_pt = float(np.nanmean(r_pool[prim]))
        new_pt = float(np.nanmean(r_new[prim]))
        pool_ci = _ci(
            boot_weighted_mean(r_pool[prim], pa.ca[prim], pa.cb[prim], pa.dyad[prim], mult)
        )
        new_ci = _ci(boot_weighted_mean(r_new[prim], pa.ca[prim], pa.cb[prim], pa.dyad[prim], mult))
        overlap = _ci_overlap(pool_ci, new_ci)
        if overlap is False:
            nonoverlap_axes.append(axis)
        per_axis[axis] = {
            "r100_pooled": pool_pt,
            "r100_pooled_ci95": pool_ci,
            "r100_new_only_projected": new_pt,
            "r100_new_only_ci95": new_ci,
            "ci_overlap": overlap,
            "n_primary_pairs": int(prim.size),
        }
    b_fail = b_pass is False
    would_fire = b_fail or bool(nonoverlap_axes)
    fired = would_fire and not cfg.smoke
    trigger = []
    if b_fail:
        trigger.append("provenance_check_b_ci_nonoverlap")
    if nonoverlap_axes:
        trigger.append(f"r100_ci_nonoverlap:{','.join(nonoverlap_axes)}")
    return {
        "primary_estimator": "new_only" if fired else "pooled",
        "per_axis": per_axis,
        "new_only_projection": f"45/45 split-half -> r{new_cols} (2r/(1+r)) -> r100 via r1",
        "fallback": {
            "criterion": K100_FALLBACK_CRITERION,
            "fired": fired,
            "would_fire": would_fire,
            "trigger": trigger or None,
            "smoke_demoted": bool(cfg.smoke and would_fire),
        },
        "bootstrap": {"B": cfg.b_boot, "seed": [BOOT_SEED, 101], "scheme": "carrier-clustered"},
    }


def k100_r_of_k(cfg: CfgPE, st: Stores, pa: PairArrays) -> dict:
    """Measured r(K) subsample curve at K in K100_R_OF_K vs the Spearman-Brown
    projection from the committed parent r10 (plan v8 §3b, exploratory).

    Per K: a seeded random K-subset of each context's valid draws, split-half
    + step-up (= the r_K estimate at that pool size); per-axis mean over ALL
    primary pairs. K > realized valid draws degrades to all valid draws
    (realized mean pool size reported)."""
    views = build_axis_views(pa, len(st.carriers))
    n_ctx, k_max = st.draw_valid.shape
    measured: dict[str, dict[str, float]] = {axis: {} for axis in views}
    realized: dict[str, float] = {}
    for k_target in K100_R_OF_K:
        rng = np.random.default_rng([SPLIT_SEED, 31337, k_target])
        scores = rng.random((n_ctx, k_max))
        scores[~st.draw_valid] = np.inf
        order = np.argsort(scores, axis=1)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(np.arange(k_max), (n_ctx, k_max)).copy(), 1)
        keep = (ranks < k_target) & st.draw_valid
        rel_k = split_half_stats(replace(st, draw_valid=keep), pa, cfg.n_splits)
        realized[str(k_target)] = float(keep.sum(axis=1).mean())
        for axis, view in views.items():
            measured[axis][str(k_target)] = float(np.nanmean(rel_k["r_full"][view.primary_idx]))
    projected = {
        axis: {str(k): float(sb_project(t["r10_mean"], K100_DRAW_OFFSET, k)) for k in K100_R_OF_K}
        for axis, t in K100_BRIDGE_TARGETS.items()
    }
    return {
        "k_grid": list(K100_R_OF_K),
        "measured_r_by_axis": measured,
        "projected_from_committed_r10": projected,
        "realized_mean_pool_size": realized,
        "subsample_seed": [SPLIT_SEED, 31337],
        "n_splits": cfg.n_splits,
        "note": "measured = draw-subsampled split-half + step-up over ALL primary pairs; "
        "projected = Spearman-Brown from the committed parent r10 (plan v8 §3b)",
    }


def k100_fire_recompute(cfg: CfgPE, st: Stores, fire_parent: dict) -> tuple[dict, dict, dict]:
    """PRIMARY fire gate recomputed from the POOLED anchors at the realized
    denominator — 12 carriers x 100 draws = 1,200 checks/value in production
    (plan v8 K6) — via the SHIPPED judge instruments (issue2564_judge). The
    committed parent 120-check table stays the COMPANION; a fired-set change
    is REPORTED loudly, never silently applied (plan v8 §4).

    Returns (fire_doc, fire_tables, companion_block); writes
    ``manipulation_check_k100.json`` beside the round outputs."""
    import issue2564_judge as JD  # sibling script (sys.path carries scripts/)

    values = BK.load_values()
    texts: dict[tuple[str, int], str] = {}
    new_draws: set[int] = set()
    n_rows_by_source: dict[str, int] = {}
    for source in ("parent", "round"):
        p = resolve_input(cfg, "raw_completions/anchors/anchors_user_fact.jsonl", source=source)
        n_rows = 0
        for line in p.open(encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            d = int(r["draw"])
            if source == "parent":
                assert d < K100_DRAW_OFFSET, (d, "parent anchors carry draw ids >= offset")
            else:
                assert d >= K100_DRAW_OFFSET, (d, "round anchors carry parent draw ids")
                new_draws.add(d)
            key = (r["context_id"], d)
            if key in texts:
                # r1 blocker k100-draw-grid-completeness: a duplicate anchor row
                # must RAISE, never last-wins into the fire denominator.
                raise RuntimeError(
                    f"[k100-fire] duplicate anchor (context_id, draw) row {key!r} in "
                    f"{source} anchors_user_fact.jsonl"
                )
            texts[key] = r["text"]
            n_rows += 1
        n_rows_by_source[source] = n_rows
    draws_re = tuple(range(K100_DRAW_OFFSET)) + tuple(sorted(new_draws))
    carriers_re = tuple(st.carriers)
    if not cfg.smoke:
        assert draws_re == tuple(range(K100_DRAWS_TOTAL)), sorted(new_draws)[:5]
        assert carriers_re == tuple(BK.CARRIER_IDS), carriers_re
    specs = [
        s
        for s in JD.programmatic_specs(values, carriers=carriers_re, draws=draws_re)
        if s["axis"] == "user_fact"
    ]
    value_rows = JD.programmatic_fire_table(specs, texts, carriers_re, draws_re)
    axis_rows = [JD.axis_summary(value_rows, "user_fact", BK.N_VALUES_PER_AXIS["user_fact"])]
    denom = len(carriers_re) * len(draws_re)
    fire_doc = {
        "value_rows": value_rows,
        "axis_rows": axis_rows,
        "meta": {
            "round": "k100",
            "smoke": cfg.smoke,
            "instrument": "programmatic name containment (issue2564_judge), user_fact only "
            "— query cells carry no compliance gate (plan v8 §9)",
            "judged_denominator": None,
            "programmatic_denominator": denom,
            "fire_threshold_pct": JD.FIRE_THRESHOLD_PCT,
            "floor_rule": ">= ceil(0.6 x width) base values fired",
            "draws": [int(draws_re[0]), int(draws_re[-1])],
            "carriers": list(carriers_re),
            "n_anchor_rows": n_rows_by_source,
            **as_metadata_dict(git_provenance(), phase="pe-analysis"),
        },
    }
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(cfg.out_dir / "manipulation_check_k100.json", _json_sanitize(fire_doc))
    # parent 120-check companion + fired-set diff (reported, never silent)
    changed: list[dict] = []
    parent_rows = {}
    for r in value_rows:
        prow = fire_parent["value_rows"].get(("user_fact", r["value_id"]))
        pv = prow["verdict"] if prow else None
        parent_rows[r["value_id"]] = pv
        if pv != r["verdict"]:
            changed.append(
                {"value_id": r["value_id"], "parent_verdict": pv, "k100_verdict": r["verdict"]}
            )
    if changed:
        logger.warning(
            "[k100-fire] FIRED-SET CHANGE vs the parent 120-check table (reported, "
            "applied at the round's own K per plan v8 §11): %s",
            changed,
        )
    companion = {
        "parent_verdicts_120": parent_rows,
        "k100_verdicts_1200": {r["value_id"]: r["verdict"] for r in value_rows},
        "fired_set_changes": changed,
        "denominators": {"parent": 120, "k100": denom},
    }
    return fire_doc, fire_tables_from_doc(fire_doc), companion


def k100_answer_length_check(bank: dict, st: Stores) -> dict:
    """Provenance check (c): per-context mean answer length, parent draws vs
    new draws, summarized per cell (report-only; plan v8 K6)."""
    contexts = bank["contexts"]
    n_all = st.n_valid.astype(np.float64)
    n_new = st.n_valid_new.astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        len_parent = (st.ans_len_mean * n_all - st.ans_len_mean_new * n_new) / (n_all - n_new)
    out: dict[str, dict] = {}
    cells = np.array([contexts[cid]["cell"] for cid in st.ctx_ids])
    for cell in sorted(set(cells.tolist())):
        m = cells == cell
        p = len_parent[m]
        nw = st.ans_len_mean_new[m]
        diff = nw - p
        out[cell] = {
            "n_contexts": int(m.sum()),
            "parent_mean_tokens": float(np.nanmean(p)),
            "new_mean_tokens": float(np.nanmean(nw)),
            "mean_paired_diff_tokens": float(np.nanmean(diff)),
            "mean_abs_paired_diff_tokens": float(np.nanmean(np.abs(diff))),
        }
    return {"per_cell": out, "note": "report-only (plan v8 §4: not a fallback trigger)"}


def k100_verdicts(doc: dict, reliability_estimator: str, smoke: bool) -> dict:
    """Registered verdict lattices (plan v8 §3b), computed from the round's
    own axes block. Point estimates drive the verdicts (the parent's
    convention); a DECIDING quantity whose carrier-clustered PAIRED-bootstrap
    CI (compute_all's ``k100_deciding`` blocks) straddles its threshold is
    flagged fragile (r1 blocker k100-verdict-fragility-ci). Informational
    under --smoke."""

    def _deciding(block: dict, axis: str, key: str) -> dict:
        dec = block.get("k100_deciding", {}).get(key) if block else None
        if block and dec is None:
            raise RuntimeError(
                f"[k100] axes.{axis} missing k100_deciding.{key} — verdicts require "
                "the paired-bootstrap deciding CI (r1 blocker k100-verdict-fragility-ci)"
            )
        return dec or {"ci95": [float("nan"), float("nan")], "fragile": None}

    uf = doc["axes"].get("user_fact", {})
    c = uf.get("direction", {}).get("arm_779ce", {}).get("mean_cos_headline")
    c_ci = uf.get("direction", {}).get("arm_779ce", {}).get("ci95") or [None, None]
    r100 = uf.get("reliability", {}).get("r100_mean")
    r_ci = uf.get("reliability", {}).get("r100_ci95") or [None, None]
    dec_ratio = _deciding(uf, "user_fact", "c_over_b")

    def _fin(v: object) -> float:
        return float(v) if isinstance(v, (int, float)) and math.isfinite(v) else float("nan")

    c, r100 = _fin(c), _fin(r100)
    b = math.sqrt(r100) if math.isfinite(r100) and r100 > 0 else float("nan")
    b_ci = [
        math.sqrt(max(_fin(r_ci[0]), 0.0)) if math.isfinite(_fin(r_ci[0])) else float("nan"),
        math.sqrt(max(_fin(r_ci[1]), 0.0)) if math.isfinite(_fin(r_ci[1])) else float("nan"),
    ]
    ratio = c / b if math.isfinite(c) and math.isfinite(b) and b > 0 else float("nan")
    if not (math.isfinite(b) and b >= K100_LATTICE_B_FLOOR) or not math.isfinite(ratio):
        uf_verdict = "unresolved"
    elif ratio >= K100_LATTICE_RATIO:
        uf_verdict = "reliability-limited"
    else:
        uf_verdict = "map-direction-loss"
    fragile_b = (
        math.isfinite(b_ci[0])
        and math.isfinite(b_ci[1])
        and b_ci[0] <= K100_LATTICE_B_FLOOR <= b_ci[1]
    )
    # deciding-ratio fragility from the PAIRED bootstrap (c and r100 resampled
    # together) — the marginal c-CI / point-b form ignored b's uncertainty
    # (r1 blocker k100-verdict-fragility-ci).
    fragile_ratio = dec_ratio.get("fragile")

    qf = doc["axes"].get("query_form", {})
    surf = qf.get("surface", {}).get("observed", {})
    s_flip, s_para = _fin(surf.get("flip_norm_mean")), _fin(surf.get("para_norm_mean"))
    s_ratio = s_flip / s_para if math.isfinite(s_flip) and s_para else float("nan")
    t_ratio = _fin(qf.get("text_space", {}).get("flip_over_para_ratio"))
    dec_g = _deciding(qf, "query_form", "g")
    g = s_ratio - t_ratio
    if not math.isfinite(g):
        qf_verdict = "not_evaluable"
    elif g >= K100_DISSOC_G:
        qf_verdict = "dissociation-holds"
    else:
        qf_verdict = "dissociation-collapses"
    return {
        "informational_smoke": bool(smoke),
        "reliability_estimator": reliability_estimator,
        "injected_name": {
            "c_mean_cos_headline_arm_779ce": c,
            "c_ci95": c_ci,
            "r100_mean": r100,
            "b100": b,
            "b100_ci95_from_r_ci": b_ci,
            "c_over_b": ratio,
            "c_over_b_ci95": dec_ratio.get("ci95"),
            "thresholds": {"b_floor": K100_LATTICE_B_FLOOR, "ratio": K100_LATTICE_RATIO},
            "verdict": uf_verdict,
            "fragile": bool(fragile_b or bool(fragile_ratio)),
            "fragile_components": {
                "b_ci_straddles_floor": bool(fragile_b),
                "c_over_b_paired_ci_straddles_ratio": fragile_ratio,
            },
            "lattice": "reliability-limited <=> b>=0.55 AND c/b>=0.35; "
            "map-direction-loss <=> b>=0.55 AND c/b<0.35; unresolved otherwise",
        },
        "query_form_dissociation": {
            "state_flip_over_para": s_ratio,
            "text_flip_over_para": t_ratio,
            "g": g,
            "g_ci95": dec_g.get("ci95"),
            "threshold_g": K100_DISSOC_G,
            "verdict": qf_verdict,
            "fragile": dec_g.get("fragile"),
            "lattice": "dissociation-holds <=> g >= 0.15; dissociation-collapses otherwise",
        },
    }


# ── main analysis ──────────────────────────────────────────────────────


def compute_all(
    cfg: CfgPE,
    bank: dict,
    st: Stores,
    fire: dict,
    frozen_global: dict[str, float] | None = None,
    k100_estimator: str = "pooled",
) -> tuple[dict, list[dict], dict]:
    """All §6 reads. Returns (minpair_delta doc, perpair rows, predictions).

    ``frozen_global`` (ffr/k100) carries the parent's per-arm global slopes;
    when present the calibration family reports ratio_to_parent_global as the
    PRIMARY ratio with the round-pooled slope as companion (plan v7 §5).

    ``k100_estimator`` (k100 only): "pooled" (default) or "new_only" (the
    registered fallback fired — ``st`` is then the new-only stores twin and
    the split-half r is projected r90 -> r100 via r1; plan v8 §4)."""
    t0 = time.time()
    pa = build_pair_arrays(bank, st, cfg.smoke, is_ffr=cfg.is_ffr, is_k100=cfg.is_k100)
    n_car = len(st.carriers)
    d = st.d

    # deltas (float64) ------------------------------------------------
    obs_tail = {
        layer: st.va_tail_mean[layer][pa.a] - st.va_tail_mean[layer][pa.b] for layer in LAYERS
    }
    obs_span19 = st.va_span_mean[PRIMARY_LAYER][pa.a] - st.va_span_mean[PRIMARY_LAYER][pa.b]
    delta_text = None if st.emb_mean is None else st.emb_mean[pa.a] - st.emb_mean[pa.b]

    dev = torch.device("cpu")
    ridge_779 = resolve_ridge(cfg, cfg.ridge_779, RIDGE_779_PATH)
    ridge_1738 = resolve_ridge(cfg, cfg.ridge_1738, RIDGE_1738_PATH)
    payload_779 = load_ridge_payload(ridge_779, d, "arm_779ce")
    payload_1738 = load_ridge_payload(ridge_1738, d, "arm_1738ce")
    ridge_meta = {
        "arm_779ce": {
            "path": str(ridge_779),
            "bytes": ridge_779.stat().st_size,
            "sha256": _sha256(ridge_779),
        },
        "arm_1738ce": {
            "path": str(ridge_1738),
            "bytes": ridge_1738.stat().st_size,
            "sha256": _sha256(ridge_1738),
        },
    }
    mapped_779 = N1M.apply_map(payload_779, st.vc[PRIMARY_LAYER], dev)
    mapped_1738 = N1M.apply_map(payload_1738, st.vc[PRIMARY_LAYER], dev)
    pred = {
        "arm_779ce": mapped_779[pa.a] - mapped_779[pa.b],
        "arm_1738ce": mapped_1738[pa.a] - mapped_1738[pa.b],
        "arm_iddelta": st.vc[PRIMARY_LAYER][pa.a] - st.vc[PRIMARY_LAYER][pa.b],
    }
    pred_iddelta_twin = {
        layer: st.vc[layer][pa.a] - st.vc[layer][pa.b] for layer in LAYERS if layer != PRIMARY_LAYER
    }

    id_check = identity_cancellation_check(
        st.vc[PRIMARY_LAYER], pa.a, pa.b, np.random.default_rng([BOOT_SEED, 999])
    )
    logger.info(
        "[pe] identity-cancellation assert PASS (max_abs_err=%.3e)", id_check["max_abs_err"]
    )

    # per-pair scalars -------------------------------------------------
    cos_arm = {arm: rowwise_cos(pred[arm], obs_tail[PRIMARY_LAYER]) for arm in ARMS}
    cos_arm_span = {arm: rowwise_cos(pred[arm], obs_span19) for arm in ARMS}
    norm_obs = {layer: np.linalg.norm(obs_tail[layer], axis=1) for layer in LAYERS}
    norm_obs_span = np.linalg.norm(obs_span19, axis=1)
    norm_pred = {arm: np.linalg.norm(pred[arm], axis=1) for arm in ARMS}
    norm_text = None if delta_text is None else np.linalg.norm(delta_text, axis=1)
    dlen = st.ans_len_mean[pa.a] - st.ans_len_mean[pa.b]

    rel = split_half_stats(st, pa, cfg.n_splits)
    if cfg.is_k100:
        if k100_estimator == "new_only":
            # fallback: st is the new-only twin (nv ~ 90); project r90 -> r100
            r10 = sb_project(rel["r_full"], st.tail_draws.shape[1], K100_DRAWS_TOTAL)
            r_key, half_key = "r100", "r45_half"
            sb_note = (
                "r90 = 2*r45/(1+r45), projected r90 -> r100 via r1 "
                "(new-only REGISTERED FALLBACK, plan v8 §4)"
            )
        else:
            r10 = rel["r_full"]  # pooled 50/50 split-half stepped up = r100
            r_key, half_key = "r100", "r50_half"
            sb_note = "r100 = 2*r50 / (1 + r50) (pooled 50/50 split-half)"
    else:
        r10 = rel["r_full"]
        r_key, half_key = "r10", "r_half"
        sb_note = "r10 = 2*r5 / (1 + r5)"

    # fire masks -------------------------------------------------------
    fired = {}
    for t in FIRE_THRESHOLDS:
        fa, fb = pair_fired_mask(pa, fire, t)
        fired[t] = fa & fb

    # edit-dose pooled OLS (per arm + observed), residuals -------------
    dose = pa.changed.astype(np.float64)
    dose_fit = {}
    resid = {}
    for name, norms in {"observed": norm_obs[PRIMARY_LAYER], **norm_pred}.items():
        icpt, slope = ols_intercept_slope(norms, dose)
        dose_fit[name] = {"intercept": icpt, "slope": slope, "n": int(pa.n)}
        resid[name] = norms - (icpt + slope * dose)

    # bootstrap + null draws (shared carrier resample, seed 2215) ------
    rng_boot = np.random.default_rng([BOOT_SEED])
    idx_draws = rng_boot.integers(0, n_car, size=(cfg.b_boot, n_car))
    mult = carrier_multiplicities(idx_draws, n_car)

    views = build_axis_views(pa, n_car)
    if not cfg.smoke:
        if cfg.is_ffr:
            # the ffr bank spans only the pilot's SURVIVING axes (a subset of
            # FFR_AXES); build_pair_arrays already asserted full-bank coverage.
            unexpected = [a for a in views if a not in BK.FFR_AXES]
            assert not unexpected, f"non-ffr axes in ffr stores: {unexpected}"
        elif cfg.is_k100:
            # k100 roster = exactly the two low-reliability cells' axes (plan v8 §3a)
            assert sorted(views) == sorted(K100_AXES), (sorted(views), K100_AXES)
        else:
            missing_axes = [a for a in AXES_ALL if a not in views]
            assert not missing_axes, f"axes missing from production stores: {missing_axes}"

    def wm(vals: np.ndarray, sel: np.ndarray) -> tuple[float, list[float]]:
        pt = float(np.nanmean(vals[sel])) if sel.size else float("nan")
        if sel.size == 0:
            return pt, [float("nan"), float("nan")]
        draws = boot_weighted_mean(vals[sel], pa.ca[sel], pa.cb[sel], pa.dyad[sel], mult)
        return pt, _ci(draws)

    def _nm(vals: np.ndarray, sel: np.ndarray) -> float:
        """nanmean over a selection; an EMPTY selection (compliance-limited
        headline, r2 blocker 7) returns NaN without numpy's empty-slice
        RuntimeWarning."""
        if sel.size == 0:
            return float("nan")
        v = vals[sel]
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    def slope_draws(sel: np.ndarray, arm: str) -> np.ndarray:
        num = boot_pair_sums(
            norm_pred[arm][sel] * norm_obs[PRIMARY_LAYER][sel],
            pa.ca[sel],
            pa.cb[sel],
            pa.dyad[sel],
            mult,
        )
        den = boot_pair_sums(
            norm_obs[PRIMARY_LAYER][sel] ** 2, pa.ca[sel], pa.cb[sel], pa.dyad[sel], mult
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    all_idx = np.arange(pa.n)
    swap_idx = np.array([i for i in all_idx if pa.cls[i] == "swap"], dtype=np.int64)
    global_slope = {
        arm: through_origin_slope(norm_pred[arm], norm_obs[PRIMARY_LAYER]) for arm in ARMS
    }
    global_slope_swap = {
        arm: through_origin_slope(norm_pred[arm][swap_idx], norm_obs[PRIMARY_LAYER][swap_idx])
        for arm in ARMS
    }
    global_slope_draws = {arm: slope_draws(all_idx, arm) for arm in ARMS}
    global_slope_swap_draws = {arm: slope_draws(swap_idx, arm) for arm in ARMS}

    axes_out: dict[str, dict] = {}
    null_schemes: dict[str, str] = {}
    for k, (axis, view) in enumerate(sorted(views.items())):
        ta = time.time()
        prim = view.primary_idx
        hmask = fired[70][prim]
        # r2 blocker 7 (plan §6): the manipulation check's per-axis floor
        # verdict (axis_row.floor_met = n_fired_base >= ceil(0.6 x width))
        # gates the HEADLINE. A below-floor axis reports NULL headline fields
        # (compliance-limited) — NEVER a silent fall-back to the unfiltered
        # pair set; *_all_values companions stay populated. Zero fired pairs
        # on a floor-met axis (no_fired_pairs, tracked separately) also nulls
        # the headline: there is nothing compliant to read.
        ar = fire["axis_rows"].get(axis)
        floor_met = bool(ar["floor_met"]) if ar is not None else True
        compliance_limited = ar is not None and not floor_met
        no_fired_pairs = not bool(hmask.any())
        headline_ok = not compliance_limited and not no_fired_pairs
        head = prim[hmask] if headline_ok else np.array([], dtype=np.int64)
        null_schemes[axis] = view.null_scheme

        # fire summary
        fire_summary = {
            "axis_row": ar,
            "n_primary_pairs": int(prim.size),
            "n_headline_pairs_fired70": int(hmask.sum()),
            "floor_met": floor_met if ar is not None else None,
            "compliance_limited": compliance_limited,
            "no_fired_pairs": no_fired_pairs,
            "headline_ok": headline_ok,
            "fired_pair_counts": {str(t): int(fired[t][prim].sum()) for t in FIRE_THRESHOLDS},
        }

        # family 5: reliability ceiling (headline pairs; all-values companions)
        ceil_pt, ceil_ci = wm(r10, head)
        ceil_all_pt, ceil_all_ci = wm(r10, prim)
        rel_axis = {
            f"{half_key}_mean": _nm(rel["r_half"], head),
            f"{r_key}_mean": ceil_pt,
            f"{r_key}_ci95": ceil_ci,
            f"{r_key}_mean_all_values": ceil_all_pt,
            f"{r_key}_ci95_all_values": ceil_all_ci,
            "noise_norm_mean": _nm(rel["noise_norm"], head),
            "spearman_brown": sb_note,
        }
        suppressed = suppression_verdict(ceil_pt, ceil_ci[0], ceil_ci[1])

        # family 1: direction fidelity
        rng_null = np.random.default_rng([NULL_SEED, k])
        direction = {}
        for arm in ARMS:
            pt, ci = wm(cos_arm[arm], head)
            pt_all, ci_all = wm(cos_arm[arm], prim)
            nd = direction_null_draws(
                view, obs_tail[PRIMARY_LAYER], pred[arm], cos_arm[arm][prim], cfg.b_null, rng_null
            )
            ratio = float("nan") if suppressed else pt / ceil_pt
            controls = {}
            for cname, cidx in {
                "install": view.install_idx,
                "instruction_paraphrase": view.para_idx if axis not in QUERY_AXES else None,
                "query_paraphrase": view.para_idx if axis in QUERY_AXES else None,
                "famswap": view.famswap_idx,
            }.items():
                if cidx is not None and cidx.size:
                    cpt, cci = wm(cos_arm[arm], cidx)
                    controls[cname] = {"mean_cos": cpt, "ci95": cci, "n_pairs": int(cidx.size)}
            direction[arm] = {
                "mean_cos_headline": pt,
                "ci95": ci,
                "mean_cos_all_values": pt_all,
                "ci95_all_values": ci_all,
                "sensitivity_mean_cos": {
                    str(t): float(np.nanmean(cos_arm[arm][prim[fired[t][prim]]]))
                    if fired[t][prim].any()
                    else float("nan")
                    for t in FIRE_THRESHOLDS
                },
                "null": {
                    "scheme": view.null_scheme,
                    "mean": float(np.nanmean(nd)),
                    "q2_5": _pct(nd, 2.5),
                    "q97_5": _pct(nd, 97.5),
                    "b": cfg.b_null,
                    "seed": [NULL_SEED, k],
                    "over": "all primary-class value pairs (fire mask NOT applied to the null grid)",
                },
                "ceiling_normalized_cos": None if suppressed else ratio,
                "ceiling_suppressed": suppressed,
                "controls": controls,
            }
            if arm != "arm_iddelta":
                # map-vs-iddelta gap (r2 concern map-iddelta-gap-missing):
                # PAIRED per-pair cos difference — the shared carrier-clustered
                # bootstrap (mult) makes the CI a paired-difference CI.
                gpt, gci = wm(cos_arm[arm] - cos_arm["arm_iddelta"], head)
                gpt_all, gci_all = wm(cos_arm[arm] - cos_arm["arm_iddelta"], prim)
                direction[arm]["gap_vs_iddelta"] = {
                    "mean_cos_gap_headline": gpt,
                    "ci95": gci,
                    "mean_cos_gap_all_values": gpt_all,
                    "ci95_all_values": gci_all,
                    "paired": "per-pair cos difference under the SHARED carrier bootstrap",
                }

        # family 2: calibration (headline floor-gated; *_all_values companions
        # for the ratio + CI reads too — r3 concern all-values-companions-incomplete —
        # under the SAME carrier-clustered bootstrap draws).
        # Pool-size-honest key names under ffr (r1 codex nit
        # ffr-round-pooled-legacy-labels): the parent keys embed the PARENT
        # pool sizes (2,778 all-pairs / 864 swaps) which the ffr round never
        # has (<=792 / <=252), so the ffr round-pooled keys are named
        # *_round_pooled / *_round_swap instead. Parent keys byte-unchanged.
        if cfg.is_ffr or cfg.is_k100:
            # pool-size-honest names for BOTH rounds (k100 pools 474 pairs, never
            # the parent 2,778/864 the parent key names embed)
            k_pool_slope, k_pool_ratio = "global_slope_round_pooled", "ratio_to_round_pooled"
            k_swap_slope, k_swap_ratio = "global_slope_round_swap", "ratio_to_round_swap"
        else:
            k_pool_slope, k_pool_ratio = "global_slope_all2778", "ratio_to_global"
            k_swap_slope, k_swap_ratio = "global_slope_swap864", "ratio_to_global_swap864"
        calibration = {}
        for arm in ARMS:
            ax_pt = through_origin_slope(norm_pred[arm][head], norm_obs[PRIMARY_LAYER][head])
            ax_all_pt = through_origin_slope(norm_pred[arm][prim], norm_obs[PRIMARY_LAYER][prim])
            ax_draws = slope_draws(head, arm)
            ax_all_draws = slope_draws(prim, arm)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio_draws = ax_draws / global_slope_draws[arm]
                ratio_swap_draws = ax_draws / global_slope_swap_draws[arm]
                ratio_all_draws = ax_all_draws / global_slope_draws[arm]
                ratio_swap_all_draws = ax_all_draws / global_slope_swap_draws[arm]
            calibration[arm] = {
                "axis_slope": ax_pt,
                "axis_slope_ci95": _ci(ax_draws),
                "axis_slope_all_values": ax_all_pt,
                "axis_slope_ci95_all_values": _ci(ax_all_draws),
                k_pool_slope: global_slope[arm],
                k_pool_ratio: ax_pt / global_slope[arm] if global_slope[arm] else float("nan"),
                f"{k_pool_ratio}_ci95": _ci(ratio_draws),
                f"{k_pool_ratio}_all_values": (
                    ax_all_pt / global_slope[arm] if global_slope[arm] else float("nan")
                ),
                f"{k_pool_ratio}_ci95_all_values": _ci(ratio_all_draws),
                k_swap_slope: global_slope_swap[arm],
                k_swap_ratio: (
                    ax_pt / global_slope_swap[arm] if global_slope_swap[arm] else float("nan")
                ),
                f"{k_swap_ratio}_ci95": _ci(ratio_swap_draws),
                f"{k_swap_ratio}_all_values": (
                    ax_all_pt / global_slope_swap[arm] if global_slope_swap[arm] else float("nan")
                ),
                f"{k_swap_ratio}_ci95_all_values": _ci(ratio_swap_all_draws),
            }
            if frozen_global is not None:
                # ffr (plan v7 §5): the PRIMARY ratio denominator is the
                # parent's FROZEN per-arm global slope; the round-pooled
                # global_slope_round_pooled keys above (round pool, <=792
                # pairs) stay populated as the companion. The denominator is a
                # fixed constant, so the CI comes from the axis-slope draws
                # alone.
                fg = frozen_global[arm]
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratio_frozen_draws = ax_draws / fg
                    ratio_frozen_all_draws = ax_all_draws / fg
                calibration[arm].update(
                    {
                        "global_slope_parent_frozen": fg,
                        "ratio_to_parent_global": ax_pt / fg if fg else float("nan"),
                        "ratio_to_parent_global_ci95": _ci(ratio_frozen_draws),
                        "ratio_to_parent_global_all_values": (
                            ax_all_pt / fg if fg else float("nan")
                        ),
                        "ratio_to_parent_global_ci95_all_values": _ci(ratio_frozen_all_draws),
                        "primary_denominator": "global_slope_parent_frozen "
                        "(parent's realized global slope; round-pooled "
                        "global_slope_round_pooled is the companion)",
                    }
                )

        # family 3: axis identity (carrier-mean per vp; headline = fired-vp
        # subset under the floor gate; all-values companions always reported)
        identity = {}
        if view.primary_grid is not None:
            # a grid row shares ONE value pair across carriers, so the pair-level
            # fire mask is constant along the row — read it off carrier 0.
            vp_fired = fired[70][view.primary_grid[:, 0]]
            grid_head = view.primary_grid[vp_fired] if (headline_ok and vp_fired.any()) else None
            med_head_draws: dict[str, np.ndarray | None] = {}
            med_all_draws: dict[str, np.ndarray] = {}
            for arm in ARMS:
                pt_rows, _, med_draws = carrier_mean_cos_median(
                    view.primary_grid, None, obs_tail[PRIMARY_LAYER], pred[arm], mult
                )
                med_all_draws[arm] = med_draws
                if grid_head is not None:
                    pt_rows_h, _, med_draws_h = carrier_mean_cos_median(
                        grid_head, None, obs_tail[PRIMARY_LAYER], pred[arm], mult
                    )
                    med_h, ci_h = float(np.nanmedian(pt_rows_h)), _ci(med_draws_h)
                else:
                    med_draws_h = None
                    med_h, ci_h = float("nan"), [float("nan"), float("nan")]
                med_head_draws[arm] = med_draws_h
                identity[arm] = {
                    "per_vp_cos": {v: float(c) for v, c in zip(view.primary_vps, pt_rows)},
                    "per_vp_fired70": {v: bool(f) for v, f in zip(view.primary_vps, vp_fired)},
                    "median": med_h,
                    "median_ci95": ci_h,
                    "median_all_values": float(np.nanmedian(pt_rows)),
                    "median_ci95_all_values": _ci(med_draws),
                    "pc1_identity_cos_exploratory": pc1_identity_cos(
                        obs_tail[PRIMARY_LAYER], pred[arm], view.primary_grid
                    ),
                }
            for arm in ARMS:
                if arm == "arm_iddelta":
                    continue
                da_, di_ = med_head_draws[arm], med_head_draws["arm_iddelta"]
                if da_ is not None and di_ is not None:
                    # r2 concern map-iddelta-gap-missing: paired median-draw
                    # difference under the SHARED carrier bootstrap.
                    identity[arm]["median_gap_vs_iddelta"] = {
                        "gap": identity[arm]["median"] - identity["arm_iddelta"]["median"],
                        "ci95": _ci(da_ - di_),
                        "paired": "median-draw difference under the SHARED carrier bootstrap",
                    }
                else:
                    identity[arm]["median_gap_vs_iddelta"] = None
                # r3 concern all-values-companions-incomplete: below-floor
                # companion — always populated (same shared-carrier bootstrap),
                # never replacing the null headline gap above.
                identity[arm]["median_gap_vs_iddelta_all_values"] = {
                    "gap": (
                        identity[arm]["median_all_values"]
                        - identity["arm_iddelta"]["median_all_values"]
                    ),
                    "ci95": _ci(med_all_draws[arm] - med_all_draws["arm_iddelta"]),
                    "paired": "median-draw difference under the SHARED carrier bootstrap",
                }
        else:
            identity = {
                "n/a": "query_content pairs are carrier dyads — no carrier-replicated "
                "value pair exists, so the carrier-mean identity read is undefined"
            }

        # family 4: cross-family aspect consistency (instruction axes only)
        cross_family: dict = {}
        if view.famswap_grid is not None and view.primary_grid is not None:
            rng_cf = np.random.default_rng([NULL_SEED, 500 + k])
            # r2 blocker 7 sweep: headline requires BOTH families' pair fired.
            vp_fired_cf = fired[70][view.primary_grid[:, 0]] & fired[70][view.famswap_grid[:, 0]]
            cf_head_ok = headline_ok and bool(vp_fired_cf.any())
            spaces = {"observed": (obs_tail[PRIMARY_LAYER], obs_tail[PRIMARY_LAYER])}
            for arm in ARMS:
                spaces[arm] = (pred[arm], pred[arm])
            for space, (da, db) in spaces.items():
                pt_rows, _, med_draws = carrier_mean_cos_median(
                    view.primary_grid, view.famswap_grid, da, db, mult
                )
                if cf_head_ok:
                    pt_rows_h, _, med_draws_h = carrier_mean_cos_median(
                        view.primary_grid[vp_fired_cf],
                        view.famswap_grid[vp_fired_cf],
                        da,
                        db,
                        mult,
                    )
                    med_h, ci_h = float(np.nanmedian(pt_rows_h)), _ci(med_draws_h)
                else:
                    med_h, ci_h = float("nan"), [float("nan"), float("nan")]
                n_vp = view.primary_grid.shape[0]
                if n_vp >= 2:
                    sm = _unit(da[view.primary_grid].mean(axis=1))
                    fm = _unit(db[view.famswap_grid].mean(axis=1))
                    cmat = fm @ sm.T
                    perms = deranged_perms(n_vp, cfg.b_null, rng_cf)
                    null_draws = np.median(cmat[perms, np.arange(n_vp)[None, :]], axis=1)
                    nscheme = "class-preserving vp derangement between the two wording families"
                else:
                    signs = rng_cf.integers(0, 2, size=(cfg.b_null, 1)) * 2 - 1
                    null_draws = (signs * np.nan_to_num(pt_rows[None, :], nan=0.0)).mean(axis=1)
                    nscheme = "sign_randomization_2value (single vp — derangement undefined)"
                cross_family[space] = {
                    "per_vp_cos": {v: float(c) for v, c in zip(view.primary_vps, pt_rows)},
                    "per_vp_fired70_both_families": {
                        v: bool(f) for v, f in zip(view.primary_vps, vp_fired_cf)
                    },
                    "median": med_h,
                    "median_ci95": ci_h,
                    "median_all_values": float(np.nanmedian(pt_rows)),
                    "median_ci95_all_values": _ci(med_draws),
                    "null": {
                        "scheme": nscheme,
                        "mean": float(np.nanmean(null_draws)),
                        "q2_5": _pct(null_draws, 2.5),
                        "q97_5": _pct(null_draws, 97.5),
                        "b": cfg.b_null,
                        "over": "all primary-class value pairs (fire mask NOT applied "
                        "to the null grid)",
                    },
                }
        else:
            cross_family = {"n/a": "no paraphrase-family swap class for this axis"}

        # family 6: text third space (observed only; headline floor-gated,
        # all-values companions — r2 blocker 7 sweep)
        para_head = (
            view.para_idx[fired[70][view.para_idx]]
            if headline_ok and view.para_idx.size
            else np.array([], dtype=np.int64)
        )
        if norm_text is None:
            # ffr round: no Qwen3 answer-embedding capture (plan v7 §5) —
            # the family is emitted as not_collected, never silently omitted.
            text_space = {
                "status": "not_collected",
                "note": "ffr round collects no Qwen3-Embedding-8B answer embeddings; "
                "text third-space reads are not_collected by design (plan v7 §5)",
            }
        else:
            flip_txt_pt, flip_txt_ci = wm(norm_text, head)
            flip_all_pt, flip_all_ci = wm(norm_text, prim)
            para_txt_pt, para_txt_ci = wm(norm_text, para_head)
            para_all_pt, para_all_ci = wm(norm_text, view.para_idx)
            text_space = {
                "flip_norm_mean": flip_txt_pt,
                "flip_norm_ci95": flip_txt_ci,
                "flip_norm_mean_all_values": flip_all_pt,
                "flip_norm_ci95_all_values": flip_all_ci,
                "paraphrase_null_norm_mean": para_txt_pt,
                "paraphrase_null_norm_ci95": para_txt_ci,
                "paraphrase_null_norm_mean_all_values": para_all_pt,
                "paraphrase_null_norm_ci95_all_values": para_all_ci,
                "flip_over_para_ratio": (
                    flip_txt_pt / para_txt_pt
                    if para_txt_pt and np.isfinite(para_txt_pt)
                    else float("nan")
                ),
                "flip_over_para_ratio_all_values": (
                    flip_all_pt / para_all_pt
                    if para_all_pt and np.isfinite(para_all_pt)
                    else float("nan")
                ),
                "note": "Qwen3-Embedding-8B mean answer embeddings (means of L2-normalized "
                "per-draw rows, NOT re-normalized); observed only — no predicted arm exists "
                "in text space",
            }
            if view.primary_grid is not None:
                vp_fired_txt = fired[70][view.primary_grid[:, 0]]
                pt_rows = np.array(
                    [
                        offdiag_pairmean_cos(delta_text[view.primary_grid[v]])
                        for v in range(len(view.primary_vps))
                    ]
                )
                cons_draws = boot_pairmean_cos_median(view.primary_grid, delta_text, idx_draws)
                if headline_ok and vp_fired_txt.any():
                    cons_draws_h = boot_pairmean_cos_median(
                        view.primary_grid[vp_fired_txt], delta_text, idx_draws
                    )
                    med_h_txt = float(np.nanmedian(pt_rows[vp_fired_txt]))
                    ci_h_txt = _ci(cons_draws_h)
                else:
                    med_h_txt, ci_h_txt = float("nan"), [float("nan"), float("nan")]
                text_space["cross_carrier_consistency"] = {
                    "per_vp_mean_pairwise_cos": {
                        v: float(c) for v, c in zip(view.primary_vps, pt_rows)
                    },
                    "median": med_h_txt,
                    "median_ci95": ci_h_txt,
                    "median_all_values": float(np.nanmedian(pt_rows)),
                    "median_ci95_all_values": _ci(cons_draws),
                }
            else:
                text_space["cross_carrier_consistency"] = None

        # family 7: surface sensitivity (descriptive; headline floor-gated,
        # all-values companions — r2 blocker 7 sweep) + edit-dose companion
        surface = {}
        for name, norms in {"observed": norm_obs[PRIMARY_LAYER], **norm_pred}.items():
            fpt, fci = wm(norms, head)
            ppt, pci = wm(norms, para_head)
            fpt_all, fci_all = wm(norms, prim)
            ppt_all, pci_all = wm(norms, view.para_idx)
            rf, _ = wm(resid[name], head)
            rp, _ = wm(resid[name], para_head)
            rf_all, _ = wm(resid[name], prim)
            rp_all, _ = wm(resid[name], view.para_idx)
            if head.size and para_head.size:
                gap_draws = boot_weighted_mean(
                    norms[head], pa.ca[head], pa.cb[head], pa.dyad[head], mult
                ) - boot_weighted_mean(
                    norms[para_head],
                    pa.ca[para_head],
                    pa.cb[para_head],
                    pa.dyad[para_head],
                    mult,
                )
                gap_ci = _ci(gap_draws)
            else:
                gap_ci = [float("nan"), float("nan")]
            gap_all_draws = boot_weighted_mean(
                norms[prim], pa.ca[prim], pa.cb[prim], pa.dyad[prim], mult
            ) - boot_weighted_mean(
                norms[view.para_idx],
                pa.ca[view.para_idx],
                pa.cb[view.para_idx],
                pa.dyad[view.para_idx],
                mult,
            )
            surface[name] = {
                "flip_norm_mean": fpt,
                "flip_norm_ci95": fci,
                "para_norm_mean": ppt,
                "para_norm_ci95": pci,
                "flip_norm_mean_all_values": fpt_all,
                "flip_norm_ci95_all_values": fci_all,
                "para_norm_mean_all_values": ppt_all,
                "para_norm_ci95_all_values": pci_all,
                "gap": fpt - ppt,
                "gap_ci95": gap_ci,
                "gap_all_values": fpt_all - ppt_all,
                "gap_ci95_all_values": _ci(gap_all_draws),
                "edit_dose_ols": dose_fit[name],
                "residualized_gap": rf - rp,
                "residualized_gap_all_values": rf_all - rp_all,
                "labeling": "DESCRIPTIVE only (plan §6: a surface-transducing map "
                "predicts a NEGATIVE gap; no H-label keys on this gap)",
            }

        # answer-length deltas (convention 6; from the capture index, gate-4
        # asserted equal to the generation-side counts)
        answer_length = {}
        for cname, cidx in {
            "primary": prim,
            "paraphrase": view.para_idx,
            "install": view.install_idx,
            "famswap": view.famswap_idx,
        }.items():
            if cidx is not None and cidx.size:
                answer_length[cname] = {
                    "mean_delta_tokens": float(np.nanmean(dlen[cidx])),
                    "mean_abs_delta_tokens": float(np.nanmean(np.abs(dlen[cidx]))),
                }

        # layer twins (arm_iddelta only — the frozen ridge maps are L19 by
        # construction) + span-pooling twin (point estimates; *_all_values
        # companions match the twins' own point-estimate convention — r3
        # concern all-values-companions-incomplete)
        layer_twins = {}
        for layer in (14, 26):
            c = rowwise_cos(pred_iddelta_twin[layer], obs_tail[layer])
            no_l = np.linalg.norm(pred_iddelta_twin[layer], axis=1)
            ax_slope = through_origin_slope(no_l[head], norm_obs[layer][head])
            ax_slope_all = through_origin_slope(no_l[prim], norm_obs[layer][prim])
            gl = through_origin_slope(no_l, norm_obs[layer])
            layer_twins[str(layer)] = {
                "arm_iddelta_mean_cos_headline": _nm(c, head),
                "arm_iddelta_mean_cos_all_values": _nm(c, prim),
                "arm_iddelta_ratio_to_global": ax_slope / gl if gl else float("nan"),
                "arm_iddelta_ratio_to_global_all_values": (
                    ax_slope_all / gl if gl else float("nan")
                ),
                "note": "iddelta only — ridge arms are L19-fit and have no twin at this layer",
            }
        span_twin = {
            arm: {
                "mean_cos_headline": _nm(cos_arm_span[arm], head),
                "mean_cos_all_values": _nm(cos_arm_span[arm], prim),
                "axis_slope": through_origin_slope(norm_pred[arm][head], norm_obs_span[head]),
                "axis_slope_all_values": through_origin_slope(
                    norm_pred[arm][prim], norm_obs_span[prim]
                ),
            }
            for arm in ARMS
        }

        # k100 deciding-statistic CIs (plan v8 §3b; r1 blocker
        # k100-verdict-fragility-ci): PAIRED per-draw ratios under the SHARED
        # carrier resample (mult) — c and r100 resample TOGETHER for c/b, and
        # all four ratio inputs resample together for the dissociation gap g.
        # Parent/ffr axis blocks gain no key (additive contract).
        k100_deciding: dict | None = None
        if cfg.is_k100 and axis == "user_fact":
            if head.size:
                c_draws = boot_weighted_mean(
                    cos_arm["arm_779ce"][head], pa.ca[head], pa.cb[head], pa.dyad[head], mult
                )
                r_draws = boot_weighted_mean(
                    r10[head], pa.ca[head], pa.cb[head], pa.dyad[head], mult
                )
                with np.errstate(invalid="ignore", divide="ignore"):
                    b_draws = np.sqrt(np.clip(r_draws, 0.0, None))
                    ratio_draws = np.where(b_draws > 0, c_draws / b_draws, np.nan)
            else:
                ratio_draws = np.full(mult.shape[0], np.nan)
            k100_deciding = {"c_over_b": k100_deciding_ci(ratio_draws, K100_LATTICE_RATIO)}
        elif cfg.is_k100 and axis == "query_form":
            if head.size and para_head.size and norm_text is not None:
                s_f = boot_weighted_mean(
                    norm_obs[PRIMARY_LAYER][head], pa.ca[head], pa.cb[head], pa.dyad[head], mult
                )
                s_p = boot_weighted_mean(
                    norm_obs[PRIMARY_LAYER][para_head],
                    pa.ca[para_head],
                    pa.cb[para_head],
                    pa.dyad[para_head],
                    mult,
                )
                t_f = boot_weighted_mean(
                    norm_text[head], pa.ca[head], pa.cb[head], pa.dyad[head], mult
                )
                t_p = boot_weighted_mean(
                    norm_text[para_head],
                    pa.ca[para_head],
                    pa.cb[para_head],
                    pa.dyad[para_head],
                    mult,
                )
                with np.errstate(invalid="ignore", divide="ignore"):
                    g_draws = s_f / s_p - t_f / t_p
                g_draws = np.where(np.isfinite(g_draws), g_draws, np.nan)
            else:
                g_draws = np.full(mult.shape[0], np.nan)
            k100_deciding = {"g": k100_deciding_ci(g_draws, K100_DISSOC_G)}

        axes_out[axis] = {
            "axis": axis,
            "primary_class": view.primary_class,
            "para_class": view.para_class,
            "n_primary_pairs": int(prim.size),
            "fire": fire_summary,
            "direction": direction,
            "calibration": calibration,
            "identity": identity,
            "cross_family": cross_family,
            "reliability": rel_axis,
            "text_space": text_space,
            "surface": surface,
            "answer_length": answer_length,
            "layer_twins": layer_twins,
            "pooling_twin_span": span_twin,
            **({"k100_deciding": k100_deciding} if k100_deciding is not None else {}),
        }
        print(
            f"[pe] axis {k + 1}/{len(views)} {axis} elapsed={time.time() - ta:.1f}s",
            flush=True,
        )

    # kNN delta retrieval ----------------------------------------------
    retrieval: dict = {"global": {}, "per_axis": {}}
    pool = obs_tail[PRIMARY_LAYER]
    for arm in ARMS:
        retrieval["global"][arm] = {
            metric: knn_retrieval(pred[arm], pool, ks=(1, 5, 10), metric=metric, pool=pool)
            for metric in ("cosine", "euclidean")
        }
    for axis, view in sorted(views.items()):
        sel = view.primary_idx
        ks = tuple(k_ for k_ in (1, 5, 10) if k_ <= sel.size)
        if not ks:
            continue
        retrieval["per_axis"][axis] = {
            arm: {
                metric: knn_retrieval(
                    pred[arm][sel], pool[sel], ks=ks, metric=metric, pool=pool[sel]
                )
                for metric in ("cosine", "euclidean")
            }
            for arm in ARMS
        }

    # per-pair rows ------------------------------------------------------
    fa70, fb70 = pair_fired_mask(pa, fire, 70)
    # r3 concern headline-pair-floor-mislabel: pair_fired_70 keeps the raw
    # both-endpoints-fired read; in_headline_70 ADDITIONALLY requires the
    # axis's floor verdict (fire.headline_ok), so figure consumers never
    # label pairs on a compliance-limited axis as headline pairs. An axis
    # without a computed view keeps the floor_met-defaults-True convention.
    headline_ok_by_axis = {ax: bool(axes_out[ax]["fire"]["headline_ok"]) for ax in axes_out}
    perpair: list[dict] = []
    for i in range(pa.n):
        perpair.append(
            {
                "pair_id": pa.ids[i],
                "pair_class": pa.cls[i],
                "axis": pa.axis[i],
                "carrier": pa.carrier_str[i],
                "value_a": pa.value_a[i],
                "value_b": pa.value_b[i],
                "orientation": pa.orientation[i],
                "changed_tokens": int(pa.changed[i]),
                "n_draws_a": int(st.n_valid[pa.a[i]]),
                "n_draws_b": int(st.n_valid[pa.b[i]]),
                "ans_len_delta": float(dlen[i]),
                "norm_obs_tail_L19": float(norm_obs[PRIMARY_LAYER][i]),
                "norm_obs_span_L19": float(norm_obs_span[i]),
                "norm_text": float(norm_text[i]) if norm_text is not None else None,
                "cos": {arm: float(cos_arm[arm][i]) for arm in ARMS},
                "cos_span": {arm: float(cos_arm_span[arm][i]) for arm in ARMS},
                "norm_pred": {arm: float(norm_pred[arm][i]) for arm in ARMS},
                half_key: float(rel["r_half"][i]),
                r_key: float(r10[i]),
                "noise_norm": float(rel["noise_norm"][i]),
                "fired_a_70": bool(fa70[i]),
                "fired_b_70": bool(fb70[i]),
                "pair_fired_70": bool(fa70[i] and fb70[i]),
                "in_headline_70": bool(
                    fa70[i] and fb70[i] and headline_ok_by_axis.get(pa.axis[i], True)
                ),
            }
        )

    contract = {
        "primary_pair_classes": {
            dv: {
                "instruction_axes": ["swap"],
                "query_axes": ["query_content", "query_form"],
                "controls": ["install", "instruction_paraphrase", "query_paraphrase"],
                "cross_family_read_class": ["famswap"],
            }
            for dv in (
                "direction_fidelity",
                "magnitude_calibration",
                "axis_identity",
                "cross_family_consistency",
                "reliability_ceiling",
                "text_third_space",
                "surface_sensitivity",
                "knn_delta_retrieval",
            )
        },
        "null_scheme": null_schemes,
        "bootstrap": {
            "scheme": "carrier-clustered (resample the 12 carrier clusters with replacement)",
            "query_content": DYADIC_BOOTSTRAP_CONVENTION,
            "B": cfg.b_boot,
            "seed": BOOT_SEED,
            "gsmall_caveat": "G=12 clusters — percentile CIs undercover (plan §6 convention 8)",
        },
        "null": {"B": cfg.b_null, "seed": NULL_SEED},
        "draw_to_pair_aggregation": DRAW_TO_PAIR_AGGREGATION,
        "orientation_conventions": ORIENTATION_CONVENTIONS,
        "split_half": {
            "n_splits": cfg.n_splits,
            "seed": SPLIT_SEED,
            "step_up": (
                f"Spearman-Brown {sb_note}" if cfg.is_k100 else "Spearman-Brown r10 = 2*r5/(1+r5)"
            ),
            "n_pairs_insufficient_draws": rel["n_pairs_insufficient_draws"],
        },
        "compliance": {
            "headline_threshold_pct": 70,
            "sensitivity_pcts": [50, 90],
            "rule": "headline per-axis reads are FLOOR-GATED: an axis whose "
            "manipulation-check floor verdict (axis_row.floor_met, "
            "n_fired_base >= ceil(0.6 x width)) is not met is compliance-limited and "
            "reports NULL headline fields; on floor-met axes the headline uses pairs "
            "whose BOTH endpoint values fired at 70; non-fired values stay in the "
            "artifact (hollow) via *_all_values companions, excluded from the headline",
        },
    }

    # round-only meta additions (the parent doc's key set stays byte-identical)
    round_meta: dict = {}
    if cfg.is_ffr or cfg.is_k100:
        assert frozen_global is not None, "round requires the parent frozen slopes"
        round_meta["round"] = "ffr" if cfg.is_ffr else "k100"
        round_meta["frozen_global_slope"] = {
            "source": str(cfg.parent_delta),
            "per_arm": dict(frozen_global),
            "note": "parent's realized per-arm global slope (global_slope_all2778), "
            "the PRIMARY ratio denominator for this round "
            + ("(plan v7 §5)" if cfg.is_ffr else "(plan v8 §11, the ffr mechanism reused)"),
        }
    if cfg.is_k100:
        round_meta["reliability_estimator"] = k100_estimator
        round_meta["parent_revision"] = cfg.parent_revision
        # consistency-checker note 1 (epm:followup-consistency v1): the v_C that
        # feeds EVERY K6 pooled/bridge read is the PARENT's committed vc bank at
        # --parent-revision; the k100-recaptured v_C serves ONLY provenance
        # check (a) pod-side and is never loaded here.
        round_meta["vc_source"] = {
            "pooled_and_bridge_reads": (
                f"parent committed vc2564_bank.pt @ {cfg.parent_revision} "
                "(staged source='parent'; issue2564_analysis.load_stores_k100)"
            ),
            "k100_recaptured_vc": (
                "provenance check (a) ONLY — compared pod-side against the parent bank "
                "(issue2564_run._k100_vc_parity); never feeds a pooled or bridge read"
            ),
        }
        retrieval["note"] = (
            "restricted-pool retrieval over the 474 k100 pairs (chance = k/474); NOT "
            "comparable to the parent's 2,778-pair global read (plan v8 §6)"
        )
    doc = {
        "meta": {
            "issue": ISSUE,
            "phase": "pe_analysis",
            **round_meta,
            "smoke": cfg.smoke,
            "layer_primary": PRIMARY_LAYER,
            "layers": list(LAYERS),
            "pooling_primary": "tail_inclusive_mean",
            "pooling_twin": "span_mean",
            "n_contexts": len(st.ctx_ids),
            "n_pairs": pa.n,
            "cells": st.cells,
            "carriers": st.carriers,
            "arms": list(ARMS),
            "ridge_payloads": ridge_meta,
            "input_files": st.input_files,
            "manip_check_meta": {
                k: fire["meta"].get(k)
                for k in ("judged_denominator", "fire_threshold_pct", "floor_rule")
            },
            "identity_cancellation_assert": id_check,
            "elapsed_s": round(time.time() - t0, 1),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            **as_metadata_dict(git_provenance(), phase="pe-analysis"),
        },
        "contract": contract,
        "axes": axes_out,
        "retrieval": retrieval,
    }
    predictions = {
        "pair_ids": pa.ids,
        "delta_obs_tail_L19": torch.from_numpy(obs_tail[PRIMARY_LAYER].astype(np.float32)),
        "delta_obs_span_L19": torch.from_numpy(obs_span19.astype(np.float32)),
        **{f"delta_pred_{arm}": torch.from_numpy(pred[arm].astype(np.float32)) for arm in ARMS},
    }
    return doc, perpair, predictions


# ── io + main ──────────────────────────────────────────────────────────


def _write_json_atomic(path: Path, obj: dict) -> None:
    """Atomic JSON write via a process-unique temp (atomic_io.atomic_replace, #2336)."""
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True))


def _json_sanitize(obj):
    """NaN/inf -> None recursively (JSON round-trip safety for consumers)."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        v = obj.item()
        return None if isinstance(v, float) and not math.isfinite(v) else v
    return obj


def write_outputs(cfg: CfgPE, doc: dict, perpair: list[dict], predictions: dict) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(cfg.out_dir / cfg.delta_name, _json_sanitize(doc))
    rows = [json.dumps(_json_sanitize(r), sort_keys=True) for r in perpair]
    with atomic_replace(cfg.out_dir / cfg.perpair_name) as tmp:
        tmp.write_text("\n".join(rows) + "\n")

    cfg.pred_dir.mkdir(parents=True, exist_ok=True)
    pair_ids = predictions["pair_ids"]
    for name, tensor in predictions.items():
        if name == "pair_ids":
            continue
        dest = cfg.pred_dir / f"{name}.pt"
        # atomic_replace temp ends ".tmp" -> never matches the "*.pt" upload glob (#2336)
        with atomic_replace(dest) as tmpp:
            torch.save(
                {"issue": ISSUE, "pair_ids": pair_ids, "layer": PRIMARY_LAYER, "tensor": tensor},
                tmpp,
            )
    upload: dict = {"mode": cfg.upload}
    if cfg.upload == "hf":
        from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

        if cfg.is_ffr:
            pred_prefix = f"{cfg.hf_prefix}/analysis_tensors/{FFR_ROUND_SEG}/predictions"
        elif cfg.is_k100:
            pred_prefix = f"{cfg.hf_prefix}/analysis_tensors/{K100_ROUND_SEG}/predictions"
        else:
            pred_prefix = f"{cfg.hf_prefix}/analysis_tensors/predictions"
        res = upload_dir_sharded(
            cfg.pred_dir,
            HF_DATA_REPO,
            pred_prefix,
            shard_glob="*.pt",
            resume_skip=False,
            delete_local=False,
        )
        upload["predictions"] = {
            "uploaded": len(res.uploaded),
            "skipped_existing": len(res.skipped_existing),
            "rerouted": len(res.rerouted),
        }
    return upload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok", flush=True)
        return 0
    cfg = build_config(args)
    print(
        f"[phase=pe_analysis] round={cfg.round} smoke={cfg.smoke} in_root={cfg.in_root} "
        f"out_dir={cfg.out_dir} b_boot={cfg.b_boot} b_null={cfg.b_null} "
        f"n_splits={cfg.n_splits} upload={cfg.upload}",
        flush=True,
    )
    frozen_global: dict[str, float] | None = None
    if cfg.is_ffr or cfg.is_k100:
        assert cfg.parent_delta is not None and cfg.parent_delta.exists(), (
            f"parent minpair_delta.json missing: {cfg.parent_delta}"
        )
        frozen_global = load_parent_frozen_slopes(cfg.parent_delta)
        logger.info("[pe] frozen parent global slopes: %s", frozen_global)
    # producer↔consumer name parity (r1 blocker ffr-bank-manifest-name-mismatch):
    # the ffr basename is the SHARED BK constant run.py writes/uploads; pinned
    # by tests/test_issue2564_ffr.py::test_ffr_bank_manifest_name_parity.
    # k100 loads the PARENT manifest (984 contexts, staged at --parent-revision;
    # plan v8 §10 call-shape bind) — load_bank_manifest's parent asserts pass
    # by construction; the roster restriction happens in load_stores_k100.
    bank_rel = (
        f"manifests/{BK.FFR_BANK_MANIFEST_FILENAME}"
        if cfg.is_ffr
        else "manifests/bank2564_manifest.json"
    )
    bank_path = resolve_input(cfg, bank_rel, source="parent" if cfg.is_k100 else "round")
    bank = load_bank_manifest(bank_path, is_ffr=cfg.is_ffr)
    assert cfg.manip_check.exists(), f"manipulation check missing: {cfg.manip_check}"
    fire = load_fire(cfg.manip_check)
    if cfg.is_ffr:
        validate_fire_coverage_ffr(bank, fire)
    k100_block: dict | None = None
    k100_estimator = "pooled"
    if cfg.is_k100:
        # plan v8 K6, in order: stores -> K=10 bridge gate -> provenance checks
        # (a)/(b)/(c) -> fire recompute at the realized denominator ->
        # pooled/new-only reliability + registered fallback -> r(K) curve.
        fire_parent = fire
        st = load_stores_k100(cfg, bank)
        pa100 = build_pair_arrays(bank, st, cfg.smoke, is_k100=True)
        bridge = k100_bridge_gate(cfg, bank, st, pa100, fire_parent)
        prov, b_pass = k100_provenance_checks(cfg, st, pa100)
        prov["check_c_answer_length"] = k100_answer_length_check(bank, st)
        fire_doc, fire, fire_companion = k100_fire_recompute(cfg, st, fire_parent)
        reliab = k100_reliability_block(cfg, st, pa100, b_pass)
        r_of_k = k100_r_of_k(cfg, st, pa100)
        if reliab["fallback"]["fired"]:
            k100_estimator = "new_only"
            logger.warning(
                "[k100] REGISTERED FALLBACK FIRED (%s) — direction reads switch to "
                "new-only means; heterogeneity is a scope caveat (plan v8 §4)",
                reliab["fallback"]["trigger"],
            )
            st = k100_new_only_stores(st)
        k100_block = {
            "bridge_gate": bridge,
            "provenance_checks": prov,
            "fire_recompute": fire_companion,
            "reliability": reliab,
            "r_of_k": r_of_k,
        }
    else:
        st = load_stores(cfg, bank)
    st.input_files[Path(bank_rel).name] = {
        "path": str(bank_path),
        "bytes": bank_path.stat().st_size,
        "sha256": _sha256(bank_path),
    }
    st.input_files[cfg.manip_check.name] = {
        "path": str(cfg.manip_check),
        "bytes": cfg.manip_check.stat().st_size,
    }
    if (cfg.is_ffr or cfg.is_k100) and cfg.parent_delta is not None:
        st.input_files["parent_minpair_delta.json"] = {
            "path": str(cfg.parent_delta),
            "bytes": cfg.parent_delta.stat().st_size,
            "sha256": _sha256(cfg.parent_delta),
        }
    doc, perpair, predictions = compute_all(
        cfg, bank, st, fire, frozen_global=frozen_global, k100_estimator=k100_estimator
    )
    if cfg.is_k100:
        assert k100_block is not None
        k100_block["verdicts"] = k100_verdicts(doc, k100_estimator, cfg.smoke)
        doc["k100"] = k100_block
        logger.info(
            "[k100] verdicts: injected_name=%s query_form=%s estimator=%s",
            k100_block["verdicts"]["injected_name"]["verdict"],
            k100_block["verdicts"]["query_form_dissociation"]["verdict"],
            k100_estimator,
        )
    upload = write_outputs(cfg, doc, perpair, predictions)
    print(
        f"[pe] wrote {cfg.out_dir / cfg.delta_name} + {cfg.perpair_name} "
        f"({len(perpair)} rows) + predictions ({upload})",
        flush=True,
    )
    print("[phase=done] pe_analysis complete", flush=True)
    return 0


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_rc)
