"""Issue #2587 PE analysis — 9B minimal-pair battery + cross-model contrasts vs #2564.

Unit 5b of the pre-split build (plan v3 §4.4/§4.5/§4.6, §6 IN FULL). VM CPU,
fully vectorized: bootstrap draws reduce to (B, n_carriers) multiplicity
einsums (ONE shared carrier-resample index matrix per battery, seed 2215);
derangement nulls gather from precomputed cosine grids; the H1 test-row
bootstrap is a single (B, n_rows) multiplicity contraction. No per-pair or
per-draw Python loops.

PORT SOURCE (plan §4.4): ``scripts/issue2564_analysis.py`` at the pinned
commit ``8265bcd75f781d8e879e924de60063e536e58dcf`` (read via ``git show``,
never checked out). The numeric helpers, PairTable machinery, split-half
reliability, axis views, null schemes, and the seven per-axis read families
are ported near-verbatim and parameterized as a two-SIDE battery:

- side ``qwen35_9b`` — 13 axes (11 parent + 2 pilots), d=4096, primary layer
  L* (READ from unit 4's frozen ``map_layer_sweep.json`` — never re-argmaxed),
  arms {``arm_fresh9b`` = the frozen L* ridge payload applied via
  ``issue779_ffc_n1m_fits.apply_map``; ``arm_iddelta9b``}, iddelta-only
  sensitivity twins at {16, 22, 30}.
- side ``qwen25_7b`` — the 11 parent axes, d=3584, L19 only, arms
  {``arm_7b_matched25k`` = unit 4's persisted matched-capacity mapped bank
  (matched EXAMPLES + estimator family, plan convention 20);
  ``arm_iddelta7b``}. NO twin layers on this side: the banked 7B store's
  extra layers (14, 26) have no 9B counterpart and the 9B twins {16, 22, 30}
  have no 7B counterpart — a cross-model contrast at a twin layer is never
  constructed (plan cross-unit constraint 5).

Cross-model module (§4.6): per axis x scale-free statistic, s_7B and s_9B
side by side + a carrier-paired cross-model delta with a carrier-paired
bootstrap CI (ONE shared 12-carrier resample per draw, BOTH models evaluated
on it, B=10,000), a t-scaled (t_11) companion and a leave-one-carrier-out
jackknife range (both re-reductions of the PERSISTED per-draw matrices,
convention 15), Spearman rho per statistic with an EXACT permutation p at
n<=12 (bitmask DP over D = sum d^2; Monte-Carlo fallback on ties) and a
per-tokenizer changed_tokens partial companion (convention 16). Symmetric
fire gating: a pair non-fired on EITHER model drops from BOTH sides.

H1 lattice (plan §3, implemented as written): Delta_map = R2_9B(L*) -
R2_7B(L19) on the REALIZED SHARED ROW INTERSECTION (exact ORDERED test-id
equality asserted), layer pair FROZEN; paired TEST-ROW bootstrap of
Delta_map with per-draw deltas persisted; the three-branch verdict is
disjoint and exhaustive (CI hi < 0 => h1_consistent; CI lo > 0 =>
h1_contradicted; else h1_inconclusive — a LIVE first-class verdict, never an
error: the plan expectation range [0.68, 0.73] straddles the 7B anchor
0.7250873; convention 14).

H2 lattice (plan §3): Spearman rho over the 11 parent axes on (a) the
observed-space per-axis separation and (b) the map-arm direction cos.
PRIMARY map-arm comparison PINNED: ``arm_fresh9b`` vs ``arm_7b_matched25k``;
``ref_7b_parent`` (the parent's committed minpair_delta.json, read at a
RECORDED commit) is a SEPARATELY LABELED sensitivity read only. Every
artifact's metadata asserts the unique ``primary_h2_7b_arm``.

Outputs (plan §6.5): ``minpair_delta_2587.json``, ``perpair_2587.jsonl``
(both sides, ``model_tag`` on every row), ``crossmodel_contrasts.json``,
plus persisted per-draw matrices under ``crossmodel_perdraw/`` and per-read-
family checkpoint JSONs under ``checkpoints/`` (§9 P10 cadence). NO figures
(unit 6), NO uploads, NO pod/GPU work.

Seeds (plan §9/§11): carrier bootstrap B=10,000 seed 2215; derangement nulls
B=10,000 seed 21620 (side offsets documented in the contract); 20 split-half
splits seed 2564; H1 row bootstrap seed [2215, 2587] (documented assumption —
the plan pins the carrier/null/split seeds but not the H1 row bootstrap).
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
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + HF token BEFORE numpy/torch import (code-style.md)

import numpy as np  # noqa: E402
import torch  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue779_ffc_n1m_fits as N1M  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.experiments.issue2587 import bank2587 as B87  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2587_analysis")

ISSUE = 2587
PORT_SOURCE_PIN = "8265bcd75f781d8e879e924de60063e536e58dcf"  # issue2564_analysis.py blob source
HF_DATA_REPO = os.environ.get("EPM_2587_DATA_WRITE_REPO", "superkaiba1/explore-persona-space-data")

# HF prefixes (local-first resolution; every prefix is a CLI arg — see build_parser).
PREFIX_2587 = "issue2587_minpair"  # unit 3a/3b battery stores + embeddings
PREFIX_2564 = "issue2564_minpair"  # parent bank manifest + parent stores
PREFIX_FITS = "issue2587_q35_map"  # unit 4 ridge payloads + preds (issue2587_fits.py CLI)
PREFIX_PREDS7B = "issue2564_minpair/analysis_tensors/predictions_7b_matched"  # unit 4 P8

# Seeds + battery sizes — parent parity (plan §9/§11).
BOOT_SEED = 2215
NULL_SEED = 21620
SPLIT_SEED = 2564
H1_BOOT_SEED = (BOOT_SEED, ISSUE)  # documented assumption: plan pins no H1 row-bootstrap seed
B_BOOT_DEFAULT = 10_000
B_NULL_DEFAULT = 10_000
N_SPLITS_DEFAULT = 20
FIRE_THRESHOLDS = (50, 70, 90)  # 70 = headline (plan §6 fire rule)
# Per-side null-seed offsets so the two sides never share a derangement stream.
NULL_OFFSET = {"qwen35_9b": 0, "qwen25_7b": 2000}

# Layers / dims (pins mirror issue2587_fits.py:106ff + issue2587_battery_run.py).
H_9B = 4096
H_7B = 3584
L19 = 19
TWIN_LAYERS_9B = (16, 22, 30)
LAYERS_7B = (14, 19, 26)  # banked 7B store carries ONLY these (constraint 5)
N_LAYERS_9B = 32
EXPECTED_EMBED_ENGINE = "0.11.0"  # vLLM pin (issue2587_battery_run.py EXPECTED_EMBED_ENGINE)
LAYER_CONVENTION_SUBSTR = "captured[L] == hidden_states[L+1]"  # constraint 4

# Arms + the PINNED H2 primary (plan §4.5 — a required deliverable).
ARM_FRESH9B = "arm_fresh9b"
ARM_IDD9B = "arm_iddelta9b"
ARM_7B_MATCHED = "arm_7b_matched25k"
ARM_IDD7B = "arm_iddelta7b"
REF_7B_PARENT = "ref_7b_parent"  # sensitivity-only label — NEVER the primary
PRIMARY_H2_7B_ARM = ARM_7B_MATCHED

PILOT_LABEL = "7B side pending #2564"  # mechanical pilot label (issue2587_judge.py PILOT_LABEL)
QUERY_AXES = ("query_content", "query_form")
PILOT_AXES = ("answer_language", "query_content_oneword")
# Classes with no complete (vp x carrier) grid: parent dyads + single-carrier oneword pairs.
GRIDLESS_CLASSES = ("query_content", "query_content_oneword")

# t_{0.975, df=11} for the convention-15 t-scaled companion (G=12 carriers -> df 11).
T975_DF11 = 2.200985160082949

ORIENTATION_CONVENTIONS = {
    "install (parent instruction axes)": "E -> value (a = bare-E query context, b = value context)",
    "install (answer_language pilot)": (
        "value -> bare (a = VALUE context, b = bare context) — OPPOSITE of the parent bank "
        "convention (bank2587.build_pilot_contexts_pairs); handled PER PAIR CLASS, never "
        "globally reoriented (cross-unit constraint 1; direction cos is orientation-invariant "
        "because pred and obs deltas share each pair's a-b orientation)"
    ),
    "swap": "value_i -> value_j by plan-listed value-index order (i < j)",
    "famswap": "para(value_i) -> para(value_j), same value-index order",
    "instruction_paraphrase": "value -> its paraphrase",
    "query_content": "carrier_i -> carrier_j by carrier-index order (i < j)",
    "query_content_oneword": "question_a -> question_b (single-carrier one-word query pairs)",
    "query_form": "form_i -> form_j by form-index order E < imp < stmt",
    "query_paraphrase": "E question -> reworded question",
}

DYADIC_BOOTSTRAP_CONVENTION = (
    "query_content pairs are carrier DYADS: the bootstrap resamples the 12 carrier VERTICES "
    "with replacement and weights each edge by the product of its sampled endpoint "
    "multiplicities (edges with an unsampled endpoint get weight 0); single-carrier pairs "
    "weight by their carrier multiplicity"
)


# ── small numeric helpers (ported verbatim from the pinned parent) ─────


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
    """Split-half -> full-K-mean reliability step-up: r_K = 2 r / (1 + r)."""
    r = np.asarray(r_half, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(r > -1.0, 2.0 * r / (1.0 + r), np.nan)
    return float(out) if np.isscalar(r_half) or out.ndim == 0 else out


def suppression_verdict(ceiling_pt: float, ci_lo: float, ci_hi: float) -> bool:
    """Plan §6 convention 1: suppress the ceiling-normalized read where the
    ceiling is nonpositive OR its bootstrap CI includes zero."""
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


def loco_multiplicities(n_carriers: int) -> np.ndarray:
    """(n_car, n_car) leave-one-carrier-out weight rows (row c drops carrier c).

    Evaluating any bootstrap battery at these rows yields the 12 LOCO point
    recomputes (convention 15's jackknife range) with zero extra machinery."""
    m = np.ones((n_carriers, n_carriers), dtype=np.float64)
    np.fill_diagonal(m, 0.0)
    return m


def boot_pair_sums(
    vals: np.ndarray,
    ca: np.ndarray,
    cb: np.ndarray,
    dyad: np.ndarray,
    mult: np.ndarray,
) -> np.ndarray:
    """Per-draw weighted sums over pairs — (B, n_car) contractions, never a
    (B, n_pairs) weight matrix (port of the pinned parent)."""
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
    """Per-draw carrier-clustered weighted means; finite vals only."""
    vals = np.asarray(vals, dtype=np.float64)
    ok = np.isfinite(vals)
    num = boot_pair_sums(np.where(ok, vals, 0.0), ca, cb, dyad, mult)
    den = boot_pair_sums(ok.astype(np.float64), ca, cb, dyad, mult)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 0, out, np.nan)


def dyad_pair_weights(mult: np.ndarray, ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """(B, n_pairs) dyadic edge weights (exposed for the bootstrap-convention pin test)."""
    return mult[:, ca] * mult[:, cb]


# ── identity-cancellation assert (plan §6 mapping-baselines pair) ──────


def identity_cancellation_check(
    vc: np.ndarray,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    rng: np.random.Generator,
    n_check: int = 32,
) -> dict:
    """Numeric fitting-free assert that the iddelta arm IS the identity+
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


# ── Spearman machinery (H2 + §4.6; exact permutation p at n<=12) ───────


def _rankdata(x: np.ndarray) -> np.ndarray:
    """Average ranks (1-based), float64. NaNs must be excluded by the caller."""
    x = np.asarray(x, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(len(x), dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    # average ties
    xs = x[order]
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    return ranks


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho = Pearson correlation of average ranks (tie-safe)."""
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def exact_spearman_perm_pvalue(rho_obs: float, n: int) -> float:
    """EXACT two-sided permutation p for Spearman rho at untied ranks, via a
    bitmask DP over the null distribution of D = sum d^2 (2^n masks x n ranks
    x D values — ~10M float adds at n=11; guarded to n <= 12)."""
    assert 2 <= n <= 12, n
    dmax = n * (n * n - 1) // 3
    dp = np.zeros((1 << n, dmax + 1), dtype=np.float64)
    dp[0, 0] = 1.0
    for mask in range(1 << n):
        row = dp[mask]
        if not row.any():
            continue
        i = bin(mask).count("1")  # next position to assign
        if i == n:
            continue
        for j in range(n):
            bit = 1 << j
            if mask & bit:
                continue
            dshift = (i - j) ** 2
            dp[mask | bit][dshift:] += row[: dmax + 1 - dshift]
    counts = dp[(1 << n) - 1]
    k = n * (n * n - 1)
    rho_of_d = 1.0 - 6.0 * np.arange(dmax + 1, dtype=np.float64) / k
    hit = np.abs(rho_of_d) >= abs(rho_obs) - 1e-12
    return float(counts[hit].sum() / counts.sum())


def mc_spearman_perm_pvalue(
    x: np.ndarray, y: np.ndarray, rho_obs: float, b: int, rng: np.random.Generator
) -> float:
    """Monte-Carlo two-sided permutation p (tie-tolerant fallback; add-one)."""
    n = len(x)
    ry = _rankdata(y)
    rx = _rankdata(x)
    hits = 0
    for _ in range(b):
        perm = rng.permutation(n)
        rp = rx[perm] - rx.mean()
        ryc = ry - ry.mean()
        den = float(np.sqrt((rp**2).sum() * (ryc**2).sum()))
        r = float((rp * ryc).sum() / den) if den > 0 else 0.0
        if abs(r) >= abs(rho_obs) - 1e-12:
            hits += 1
    return (1 + hits) / (b + 1)


def spearman_block(
    x: np.ndarray, y: np.ndarray, rng: np.random.Generator, mc_b: int = 10_000
) -> dict:
    """rho + two-sided permutation p over the finite-pair subset. Exact DP p
    at n<=12 with untied ranks; Monte-Carlo (labeled) on ties or larger n."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    n = int(ok.sum())
    if n < 3:
        return {"rho": float("nan"), "n": n, "p": float("nan"), "method": "insufficient-n"}
    xs, ys = x[ok], y[ok]
    rho = spearman_rho(xs, ys)
    tied = len(np.unique(xs)) < n or len(np.unique(ys)) < n
    if not tied and n <= 12:
        return {"rho": rho, "n": n, "p": exact_spearman_perm_pvalue(rho, n), "method": "exact-dp"}
    return {
        "rho": rho,
        "n": n,
        "p": mc_spearman_perm_pvalue(xs, ys, rho, mc_b, rng),
        "method": f"monte-carlo({mc_b}){' ties' if tied else ''}",
    }


def partial_spearman(x: np.ndarray, y: np.ndarray, zx: np.ndarray, zy: np.ndarray) -> float:
    """Partial Spearman with PER-TOKENIZER covariates (convention 16): each
    side's ranks are residualized on ITS OWN tokenizer's changed_tokens ranks,
    then the residuals are Pearson-correlated."""
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(zx) & np.isfinite(zy)
    if int(ok.sum()) < 4:
        return float("nan")

    def _resid(v: np.ndarray, z: np.ndarray) -> np.ndarray:
        rv, rz = _rankdata(v), _rankdata(z)
        icpt, slope = ols_intercept_slope(rv, rz)
        return rv - (icpt + slope * rz)

    ex = _resid(x[ok], zx[ok])
    ey = _resid(y[ok], zy[ok])
    den = float(np.sqrt((ex**2).sum() * (ey**2).sum()))
    return float((ex * ey).sum() / den) if den > 0 else float("nan")


# ── config ─────────────────────────────────────────────────────────────


@dataclass
class CfgX:
    in_root_9b: Path | None
    in_root_7b: Path | None
    stage_dir: Path
    out_dir: Path
    bank_9b: Path
    bank_7b: Path | None
    manip_9b: Path
    manip_7b: Path
    sweep_json: Path
    ridge_9b: Path | None
    preds_9b: Path | None
    preds7b_dir: Path | None
    ref7b_parent: Path
    ref7b_parent_commit: str
    embed_parity_report: Path | None
    smoke: bool
    b_boot: int
    b_null: int
    n_splits: int
    prefix_2587: str
    prefix_2564: str
    prefix_fits: str
    prefix_preds7b: str

    @property
    def ckpt_dir(self) -> Path:
        return self.out_dir / "checkpoints"

    @property
    def perdraw_dir(self) -> Path:
        return self.out_dir / "crossmodel_perdraw"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0].replace("%", "%%"))
    ap.add_argument("--in-root-9b", type=Path, default=None, help="local 2587 out-root mirror")
    ap.add_argument("--in-root-7b", type=Path, default=None, help="local 2564 out-root mirror")
    ap.add_argument("--stage-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--bank-9b", type=Path, default=None, help="bank_manifest.json (unit 1)")
    ap.add_argument("--bank-7b", type=Path, default=None, help="bank2564_manifest.json override")
    ap.add_argument("--manip-9b", type=Path, default=None)
    ap.add_argument("--manip-7b", type=Path, default=None)
    ap.add_argument("--sweep-json", type=Path, default=None, help="map_layer_sweep.json (unit 4)")
    ap.add_argument("--ridge-9b", type=Path, default=None, help="local L* ridge payload override")
    ap.add_argument("--preds-9b", type=Path, default=None, help="local L*_preds.pt override")
    ap.add_argument("--preds7b-dir", type=Path, default=None, help="dir with matched-7B .pt files")
    ap.add_argument("--ref7b-parent", type=Path, default=None, help="parent minpair_delta.json")
    ap.add_argument(
        "--ref7b-parent-commit",
        default=None,
        help="the EXACT commit the ref7b-parent JSON was read at (recorded; required with it)",
    )
    ap.add_argument("--embed-parity-report", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--b-boot", type=int, default=None)
    ap.add_argument("--b-null", type=int, default=None)
    ap.add_argument("--n-splits", type=int, default=N_SPLITS_DEFAULT)
    ap.add_argument("--prefix-2587", default=PREFIX_2587)
    ap.add_argument("--prefix-2564", default=PREFIX_2564)
    ap.add_argument("--prefix-fits", default=PREFIX_FITS)
    ap.add_argument("--prefix-preds7b", default=PREFIX_PREDS7B)
    ap.add_argument("--import-check", action="store_true")
    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_config(args: argparse.Namespace) -> CfgX:
    """Resolve the CLI namespace. Smoke rebinds out-dir/B, never inputs, and
    REQUIRES explicit --manip-9b/--manip-7b (the parent's [g5] convention: a
    smoke must never silently gate on the committed PRODUCTION fire verdicts)."""
    smoke = bool(args.smoke)
    results_2587 = _REPO_ROOT / "eval_results" / "issue_2587"
    out_dir = (
        Path(args.out_dir) if args.out_dir else (results_2587 / "smoke" if smoke else results_2587)
    )
    stage_dir = (
        Path(args.stage_dir)
        if args.stage_dir
        else _REPO_ROOT
        / "data"
        / "issue_2587"
        / "hf_dl"
        / ("an_stage_smoke" if smoke else "an_stage")
    )
    if smoke and (args.manip_9b is None or args.manip_7b is None):
        raise SystemExit(
            "--manip-9b AND --manip-7b are REQUIRED under --smoke: a smoke run must never "
            "silently read the committed PRODUCTION manipulation-check fire verdicts."
        )
    manip_9b = (
        Path(args.manip_9b) if args.manip_9b else results_2587 / "manipulation_check_2587.json"
    )
    manip_7b = (
        Path(args.manip_7b)
        if args.manip_7b
        else _REPO_ROOT / "eval_results" / "issue_2564" / "manipulation_check.json"
    )
    ref7b = (
        Path(args.ref7b_parent)
        if args.ref7b_parent
        else _REPO_ROOT / "eval_results" / "issue_2564" / "minpair_delta.json"
    )
    if args.ref7b_parent is not None and not args.ref7b_parent_commit:
        raise SystemExit("--ref7b-parent-commit is REQUIRED with --ref7b-parent (plan §4.5)")
    return CfgX(
        in_root_9b=Path(args.in_root_9b) if args.in_root_9b else None,
        in_root_7b=Path(args.in_root_7b) if args.in_root_7b else None,
        stage_dir=stage_dir,
        out_dir=out_dir,
        bank_9b=Path(args.bank_9b) if args.bank_9b else results_2587 / "bank_manifest.json",
        bank_7b=Path(args.bank_7b) if args.bank_7b else None,
        manip_9b=manip_9b,
        manip_7b=manip_7b,
        sweep_json=Path(args.sweep_json)
        if args.sweep_json
        else results_2587 / "map_layer_sweep.json",
        ridge_9b=Path(args.ridge_9b) if args.ridge_9b else None,
        preds_9b=Path(args.preds_9b) if args.preds_9b else None,
        preds7b_dir=Path(args.preds7b_dir) if args.preds7b_dir else None,
        ref7b_parent=ref7b,
        ref7b_parent_commit=args.ref7b_parent_commit or "UNRECORDED — pass --ref7b-parent-commit",
        embed_parity_report=Path(args.embed_parity_report) if args.embed_parity_report else None,
        smoke=smoke,
        b_boot=int(args.b_boot) if args.b_boot is not None else (100 if smoke else B_BOOT_DEFAULT),
        b_null=int(args.b_null) if args.b_null is not None else (100 if smoke else B_NULL_DEFAULT),
        n_splits=int(args.n_splits),
        prefix_2587=args.prefix_2587,
        prefix_2564=args.prefix_2564,
        prefix_fits=args.prefix_fits,
        prefix_preds7b=args.prefix_preds7b,
    )


def resolve_rel(cfg: CfgX, in_root: Path | None, prefix: str, rel: str) -> Path:
    """``<in_root>/<rel>`` when present, else the staged copy, else stage
    ``<prefix>/<rel>`` from the HF data repo (fail loud on a missing artifact)."""
    if in_root is not None:
        cand = in_root / rel
        if cand.exists():
            return cand
    target = cfg.stage_dir / prefix / rel
    if target.exists():
        return target
    from explore_persona_space.orchestrate.hub import stage_hub_file

    logger.info("[an] staging %s/%s from %s", prefix, rel, HF_DATA_REPO)
    return Path(stage_hub_file(HF_DATA_REPO, f"{prefix}/{rel}", target))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ── frozen L* + ridge payload + engine parity ──────────────────────────


def load_lstar(sweep_json: Path) -> dict:
    """READ unit 4's frozen L* — never re-argmaxed (plan §3: the freeze is
    val-selected/test-read; re-selecting here would double-dip the test set)."""
    doc = json.loads(Path(sweep_json).read_text())
    blk = doc["lstar"]
    if not blk.get("frozen"):
        raise RuntimeError(f"lstar block is not frozen: {blk} — refuse to re-derive L*")
    lstar = int(blk["lstar"])
    assert 0 <= lstar < N_LAYERS_9B, lstar
    return {"lstar": lstar, "block": blk, "sweep_path": str(sweep_json)}


def load_ridge_payload(path: Path, expect_d: int, arm: str) -> dict:
    """Load + semantically validate a frozen ridge payload (kind/shape contract).

    ``weights_only=False`` is correct for these sha-pinned SELF-PRODUCED
    bundles (torch>=2.6 convention, feedback_torch26_weights_only)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload.get("kind") == "ridge", (arm, payload.get("kind"))
    w = payload["W"]
    assert tuple(w.shape) == (expect_d, expect_d), (arm, tuple(w.shape), expect_d)
    for k in ("xmu", "xsd", "ymu"):
        assert tuple(payload[k].shape)[-1] == expect_d, (arm, k, tuple(payload[k].shape))
    return payload


def assert_engine_parity(side: str, npz_engine: str | None, report: Path | None) -> dict:
    """Cross-unit constraint 3: embed engine-version parity is ASSERTED, not
    assumed. 9B npz MUST record vllm_version; parity passes at the repo pin or
    via an explicit passing parity report. The 7B side's banked vectors ARE
    the reference — an absent key there records by-pin provenance."""
    if npz_engine is None:
        if side == "qwen25_7b":
            return {
                "side": side,
                "engine": None,
                "mode": "reference-by-pin",
                "note": "parent banked vectors: vLLM 0.11.0 by repo-pin provenance (plan §4.4)",
            }
        raise RuntimeError(
            f"[{side}] embeddings npz records NO vllm_version — the engine-parity gate "
            "(cross-unit constraint 3) cannot pass; regenerate embeddings with unit 3b"
        )
    if str(npz_engine) == EXPECTED_EMBED_ENGINE:
        return {"side": side, "engine": str(npz_engine), "mode": "repo-pin"}
    if report is None:
        raise RuntimeError(
            f"[{side}] embed engine {npz_engine!r} != pinned {EXPECTED_EMBED_ENGINE!r} and no "
            "--embed-parity-report was supplied (cross-unit constraint 3)"
        )
    rep = json.loads(Path(report).read_text())
    assert rep.get("parity_pass") is True, ("embed parity report did not pass", rep)
    assert str(rep.get("engine")) == str(npz_engine), (rep.get("engine"), npz_engine)
    assert str(rep.get("reference_engine")) == EXPECTED_EMBED_ENGINE, rep.get("reference_engine")
    return {
        "side": side,
        "engine": str(npz_engine),
        "mode": "parity-report",
        "report": str(report),
    }


# ── side spec + stores ─────────────────────────────────────────────────


@dataclass
class SideSpec:
    name: str
    d: int
    primary_layer: int
    twin_layers: tuple[int, ...]
    store_layers: tuple[int, ...]  # layer list the va/vc stores must carry
    instruction_axes: tuple[str, ...]
    query_axes: tuple[str, ...]
    pilot_axes: tuple[str, ...]
    primary_class_by_axis: dict
    para_class_by_axis: dict
    map_arm: str
    id_arm: str
    expected_contexts: int | None
    expected_pairs: int | None

    @property
    def arms(self) -> tuple[str, ...]:
        return (self.map_arm, self.id_arm)


def make_spec_9b(lstar: int, instruction_axes: tuple[str, ...]) -> SideSpec:
    layers = tuple(sorted({lstar, *TWIN_LAYERS_9B}))
    return SideSpec(
        name="qwen35_9b",
        d=H_9B,
        primary_layer=lstar,
        twin_layers=tuple(t for t in TWIN_LAYERS_9B if t != lstar),
        store_layers=layers,
        instruction_axes=instruction_axes,
        query_axes=QUERY_AXES,
        pilot_axes=PILOT_AXES,
        primary_class_by_axis={
            **{a: "swap" for a in instruction_axes},
            "query_content": "query_content",
            "query_form": "query_form",
            "answer_language": "swap",
            "query_content_oneword": "query_content_oneword",
        },
        para_class_by_axis={
            **{a: "instruction_paraphrase" for a in instruction_axes},
            "query_content": "query_paraphrase",
            "query_form": "query_paraphrase",
            "answer_language": None,  # pilot: no paraphrase family (judge has_para=False)
            "query_content_oneword": None,
        },
        map_arm=ARM_FRESH9B,
        id_arm=ARM_IDD9B,
        expected_contexts=B87.N_CONTEXTS,
        expected_pairs=B87.N_PAIRS,
    )


def make_spec_7b(instruction_axes: tuple[str, ...]) -> SideSpec:
    return SideSpec(
        name="qwen25_7b",
        d=H_7B,
        primary_layer=L19,
        # constraint 5: NO twin layers on the 7B side — the banked extras
        # (14, 26) have no 9B counterpart; twins never enter a cross-model read.
        twin_layers=(),
        store_layers=LAYERS_7B,
        instruction_axes=instruction_axes,
        query_axes=QUERY_AXES,
        pilot_axes=(),
        primary_class_by_axis={
            **{a: "swap" for a in instruction_axes},
            "query_content": "query_content",
            "query_form": "query_form",
        },
        para_class_by_axis={
            **{a: "instruction_paraphrase" for a in instruction_axes},
            "query_content": "query_paraphrase",
            "query_form": "query_paraphrase",
        },
        map_arm=ARM_7B_MATCHED,
        id_arm=ARM_IDD7B,
        expected_contexts=B87.N_PARENT_CONTEXTS,
        expected_pairs=B87.N_PARENT_PAIRS,
    )


@dataclass
class Stores:
    ctx_ids: list
    row_of: dict
    cells: list
    carriers: list
    va_tail_mean: dict  # layer -> (n_ctx, d) float64
    va_span_mean: dict
    tail_draws: np.ndarray  # (n_ctx, k_max, d) float32, PRIMARY layer
    draw_valid: np.ndarray
    n_valid: np.ndarray
    ans_len_mean: np.ndarray
    vc: dict  # layer -> (n_ctx, d) float64
    emb_mean: np.ndarray
    emb_engine: str | None
    d: int
    exclusions: dict = field(default_factory=dict)
    input_files: dict = field(default_factory=dict)


def _store_col(store: dict, layer: int) -> int:
    """Constraint 4: resolve a layer's tensor column via the store's OWN
    ``layers`` list — never a positional assumption (the capture convention is
    ``captured[L] == hidden_states[L+1]``; unit 3b docstrings)."""
    layers = [int(x) for x in store["layers"]]
    assert layer in layers, (layer, layers)
    conv = store.get("layer_convention")
    if conv is not None:
        assert LAYER_CONVENTION_SUBSTR in str(conv), conv
    return layers.index(layer)


def _finish_stores(
    ctx_ids: list,
    row_of: dict,
    cells: list,
    carriers: list,
    layers: tuple[int, ...],
    tail_sum: dict,
    span_sum: dict,
    len_sum: np.ndarray,
    cnt: np.ndarray,
    prim_chunks: list,
    k_max: int,
    vc: dict,
    emb_mean: np.ndarray,
    emb_engine: str | None,
    d: int,
    exclusions: dict,
    files: dict,
) -> Stores:
    n_ctx = len(ctx_ids)
    zero = [ctx_ids[i] for i in range(n_ctx) if cnt[i] == 0]
    if zero:
        raise RuntimeError(f"contexts with ZERO valid (non-empty, non-leak) draws: {zero[:10]}")
    va_tail_mean = {layer: tail_sum[layer] / cnt[:, None] for layer in layers}
    va_span_mean = {layer: span_sum[layer] / cnt[:, None] for layer in layers}
    ans_len_mean = len_sum / cnt
    tail_draws = np.zeros((n_ctx, k_max, d), dtype=np.float32)
    draw_valid = np.zeros((n_ctx, k_max), dtype=bool)
    for ctx_v, draw_v, rows_v in prim_chunks:
        key = ctx_v * k_max + draw_v
        assert len(np.unique(key)) == len(key), "duplicate (context, draw) slot within a va store"
        assert not draw_valid[ctx_v, draw_v].any(), "duplicate (context, draw) slot in va stores"
        tail_draws[ctx_v, draw_v] = rows_v
        draw_valid[ctx_v, draw_v] = True
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
        emb_engine=emb_engine,
        d=d,
        exclusions=exclusions,
        input_files=files,
    )


def _load_embeddings(path: Path, ctx_ids: list, files: dict) -> tuple[np.ndarray, str | None]:
    with np.load(path, allow_pickle=False) as z:
        emb_ids = [str(x) for x in z["context_ids"].tolist()]
        emb = z["emb_mean"].astype(np.float64)
        engine = str(z["vllm_version"]) if "vllm_version" in z.files else None
    files["means_anchors.npz"] = {"path": str(path), "bytes": path.stat().st_size}
    emb_of = {cid: i for i, cid in enumerate(emb_ids)}
    missing = [cid for cid in ctx_ids if cid not in emb_of]
    assert not missing, f"contexts missing from embedding means: {missing[:5]}"
    return emb[[emb_of[cid] for cid in ctx_ids]], engine


def load_stores_9b(cfg: CfgX, bank: dict, spec: SideSpec) -> Stores:
    """Assemble the 9B side from unit 3b's per-cell va2587/vc2587 stores.

    Index key is ``rows`` (NOT the parent's ``index``); think-leak rows are
    EXCLUDED from every read (plan §4.2 — leaked rows are flagged for
    exclusion) alongside empty-completion rows; counts recorded."""
    files: dict = {}
    contexts = bank["contexts"]
    cells = sorted({c["cell"] for c in contexts.values()})
    ctx_ids: list[str] = []
    vc_rows: list[np.ndarray] = []
    layers = spec.store_layers
    per_cell: list[tuple[str, dict]] = []
    for cell in cells:
        rel_vc = f"analysis_tensors/vc2587/{cell}.pt"
        p_vc = resolve_rel(cfg, cfg.in_root_9b, cfg.prefix_2587, rel_vc)
        store = torch.load(p_vc, map_location="cpu", weights_only=False)
        files[f"vc2587_{cell}.pt"] = {"path": str(p_vc), "bytes": p_vc.stat().st_size}
        cols = [_store_col(store, layer) for layer in layers]
        vc_t = store["vc"].to(torch.float64).numpy()
        assert vc_t.ndim == 3 and vc_t.shape[2] == spec.d, vc_t.shape
        cids = [str(x) for x in store["context_ids"]]
        assert vc_t.shape[0] == len(cids), (vc_t.shape, len(cids))
        ctx_ids.extend(cids)
        vc_rows.append(np.ascontiguousarray(vc_t[:, cols, :]))
        per_cell.append((cell, store))
    row_of = {cid: i for i, cid in enumerate(ctx_ids)}
    assert len(row_of) == len(ctx_ids), "duplicate context ids across vc2587 cell stores"
    missing_bank = [cid for cid in ctx_ids if cid not in contexts]
    assert not missing_bank, f"vc contexts absent from bank manifest: {missing_bank[:5]}"
    vc_all = np.concatenate(vc_rows, axis=0)  # (n_ctx, len(layers), d)
    vc = {layer: np.ascontiguousarray(vc_all[:, k, :]) for k, layer in enumerate(layers)}
    carriers = sorted({contexts[cid]["carrier"] for cid in ctx_ids})

    n_ctx = len(ctx_ids)
    tail_sum = {layer: np.zeros((n_ctx, spec.d), dtype=np.float64) for layer in layers}
    span_sum = {layer: np.zeros((n_ctx, spec.d), dtype=np.float64) for layer in layers}
    len_sum = np.zeros(n_ctx, dtype=np.float64)
    cnt = np.zeros(n_ctx, dtype=np.int64)
    prim_chunks: list = []
    k_max = 0
    n_leak = 0
    n_empty = 0
    for cell, _vc_store in per_cell:
        rel = f"analysis_tensors/va2587/{cell}.pt"
        p = resolve_rel(cfg, cfg.in_root_9b, cfg.prefix_2587, rel)
        store = torch.load(p, map_location="cpu", weights_only=False)
        files[f"va2587_{cell}.pt"] = {"path": str(p), "bytes": p.stat().st_size}
        cols = {layer: _store_col(store, layer) for layer in layers}
        idx_rows = store["rows"]
        tail = store["va_tail_incl"].to(torch.float64).numpy()
        span = store["va_span"].to(torch.float64).numpy()
        n_rows = len(idx_rows)
        assert tail.shape == (n_rows, len(store["layers"]), spec.d), tail.shape
        ctx_idx = np.array([row_of.get(r["context_id"], -1) for r in idx_rows], dtype=np.int64)
        n_comp = np.array([int(r["n_completion_tokens"]) for r in idx_rows], dtype=np.int64)
        draw = np.array([int(r["draw"]) for r in idx_rows], dtype=np.int64)
        leak = np.array([bool(r["think_leak"]) for r in idx_rows], dtype=bool)
        empty_mask = np.zeros(n_rows, dtype=bool)
        empty_ids = np.array(sorted(int(i) for i in store.get("empty_rows", [])), dtype=np.int64)
        if empty_ids.size:
            empty_mask[empty_ids] = True
        n_absent = int((ctx_idx < 0).sum())
        assert n_absent == 0, (cell, n_absent, "va rows reference contexts absent from vc2587")
        valid = (ctx_idx >= 0) & (n_comp > 0) & ~empty_mask & ~leak
        n_leak += int(leak.sum())
        n_empty += int((empty_mask | (n_comp <= 0)).sum())
        for layer in layers:
            np.add.at(tail_sum[layer], ctx_idx[valid], tail[valid, cols[layer], :])
            np.add.at(span_sum[layer], ctx_idx[valid], span[valid, cols[layer], :])
        np.add.at(len_sum, ctx_idx[valid], n_comp[valid].astype(np.float64))
        np.add.at(cnt, ctx_idx[valid], 1)
        if valid.any():
            k_max = max(k_max, int(draw[valid].max()) + 1)
            prim_chunks.append(
                (
                    ctx_idx[valid],
                    draw[valid],
                    tail[valid, cols[spec.primary_layer], :].astype(np.float32),
                )
            )
    emb_path = resolve_rel(
        cfg,
        cfg.in_root_9b,
        cfg.prefix_2587,
        "analysis_tensors/embeddings_qwen3_8b/means_anchors.npz",
    )
    emb_mean, engine = _load_embeddings(emb_path, ctx_ids, files)
    return _finish_stores(
        ctx_ids,
        row_of,
        cells,
        carriers,
        layers,
        tail_sum,
        span_sum,
        len_sum,
        cnt,
        prim_chunks,
        k_max,
        vc,
        emb_mean,
        engine,
        spec.d,
        {"think_leak_rows_excluded": n_leak, "empty_rows_excluded": n_empty},
        files,
    )


def load_stores_7b(cfg: CfgX, bank: dict, spec: SideSpec) -> Stores:
    """Assemble the 7B side from the PARENT's stores (vc2564_bank.pt single
    file + per-cell va2564_<cell>.pt with the parent's ``index`` key)."""
    files: dict = {}
    p_vc = resolve_rel(
        cfg, cfg.in_root_7b, cfg.prefix_2564, "analysis_tensors/vc2564/vc2564_bank.pt"
    )
    vc_store = torch.load(p_vc, map_location="cpu", weights_only=False)
    files["vc2564_bank.pt"] = {"path": str(p_vc), "bytes": p_vc.stat().st_size}
    assert tuple(int(x) for x in vc_store["layers"]) == LAYERS_7B, vc_store["layers"]
    ctx_ids = list(vc_store["context_ids"])
    row_of = {cid: i for i, cid in enumerate(ctx_ids)}
    vc_t = vc_store["vc"].to(torch.float64).numpy()
    assert vc_t.shape[2] == spec.d, vc_t.shape
    cols = {layer: _store_col(vc_store, layer) for layer in spec.store_layers}
    vc = {layer: np.ascontiguousarray(vc_t[:, cols[layer], :]) for layer in spec.store_layers}
    contexts = bank["contexts"]
    missing_bank = [cid for cid in ctx_ids if cid not in contexts]
    assert not missing_bank, f"vc contexts absent from parent bank manifest: {missing_bank[:5]}"
    cells = sorted({contexts[cid]["cell"] for cid in ctx_ids})
    carriers = sorted({contexts[cid]["carrier"] for cid in ctx_ids})

    n_ctx = len(ctx_ids)
    layers = spec.store_layers
    tail_sum = {layer: np.zeros((n_ctx, spec.d), dtype=np.float64) for layer in layers}
    span_sum = {layer: np.zeros((n_ctx, spec.d), dtype=np.float64) for layer in layers}
    len_sum = np.zeros(n_ctx, dtype=np.float64)
    cnt = np.zeros(n_ctx, dtype=np.int64)
    prim_chunks: list = []
    k_max = 0
    n_empty = 0
    for cell in cells:
        rel = f"analysis_tensors/va2564/va2564_{cell}.pt"
        p = resolve_rel(cfg, cfg.in_root_7b, cfg.prefix_2564, rel)
        store = torch.load(p, map_location="cpu", weights_only=False)
        files[f"va2564_{cell}.pt"] = {"path": str(p), "bytes": p.stat().st_size}
        scols = {layer: _store_col(store, layer) for layer in layers}
        idx_rows = store["index"]
        tail = store["va_tail_incl"].to(torch.float64).numpy()
        span = store["va_span"].to(torch.float64).numpy()
        n_rows = len(idx_rows)
        ctx_idx = np.array([row_of.get(r["context_id"], -1) for r in idx_rows], dtype=np.int64)
        n_comp = np.array([int(r["n_completion_tokens"]) for r in idx_rows], dtype=np.int64)
        draw = np.array([int(r["draw"]) for r in idx_rows], dtype=np.int64)
        empty_mask = np.zeros(n_rows, dtype=bool)
        empty_ids = np.array(sorted(int(i) for i in store.get("empty_rows", [])), dtype=np.int64)
        if empty_ids.size:
            empty_mask[empty_ids] = True
        n_absent = int((ctx_idx < 0).sum())
        if n_absent:
            logger.warning(
                "[an] va2564_%s: %d/%d rows ctx-absent (dropped)", cell, n_absent, n_rows
            )
        files[f"va2564_{cell}.pt"]["n_rows_ctx_absent_from_vc"] = n_absent
        valid = (ctx_idx >= 0) & (n_comp > 0) & ~empty_mask
        n_empty += int((empty_mask | (n_comp <= 0)).sum())
        for layer in layers:
            np.add.at(tail_sum[layer], ctx_idx[valid], tail[valid, scols[layer], :])
            np.add.at(span_sum[layer], ctx_idx[valid], span[valid, scols[layer], :])
        np.add.at(len_sum, ctx_idx[valid], n_comp[valid].astype(np.float64))
        np.add.at(cnt, ctx_idx[valid], 1)
        if valid.any():
            k_max = max(k_max, int(draw[valid].max()) + 1)
            prim_chunks.append(
                (
                    ctx_idx[valid],
                    draw[valid],
                    tail[valid, scols[spec.primary_layer], :].astype(np.float32),
                )
            )
    emb_path = resolve_rel(
        cfg,
        cfg.in_root_7b,
        cfg.prefix_2564,
        "analysis_tensors/embeddings_qwen3_8b/means_anchors.npz",
    )
    emb_mean, engine = _load_embeddings(emb_path, ctx_ids, files)
    return _finish_stores(
        ctx_ids,
        row_of,
        cells,
        carriers,
        layers,
        tail_sum,
        span_sum,
        len_sum,
        cnt,
        prim_chunks,
        k_max,
        vc,
        emb_mean,
        engine,
        spec.d,
        {"think_leak_rows_excluded": 0, "empty_rows_excluded": n_empty},
        files,
    )


# ── pair table (ported; axis key handles pilots via constraint 2) ──────


@dataclass
class PairArrays:
    ids: list
    cls: list
    axis: list
    value_a: list
    value_b: list
    carrier_str: list
    a: np.ndarray
    b: np.ndarray
    ca: np.ndarray
    cb: np.ndarray
    dyad: np.ndarray
    changed: np.ndarray
    orientation: list
    n: int


def build_pair_arrays(bank: dict, st: Stores, spec: SideSpec, smoke: bool) -> PairArrays:
    """Restrict the bank's pairs to contexts present in the stores; production
    asserts FULL coverage against the spec's expected counts."""
    car_of = {c: i for i, c in enumerate(st.carriers)}
    keep = [p for p in bank["pairs"] if p["a"] in st.row_of and p["b"] in st.row_of]
    if not keep:
        raise RuntimeError("empty pair selection: no bank pair has both contexts in the stores")
    if not smoke and spec.expected_pairs is not None:
        assert len(st.ctx_ids) == spec.expected_contexts, (len(st.ctx_ids), spec.expected_contexts)
        assert len(keep) == spec.expected_pairs, (len(keep), spec.expected_pairs)

    ids, cls, axis, va_, vb_, cstr, orient = [], [], [], [], [], [], []
    a_i, b_i, ca_i, cb_i, dy, chg = [], [], [], [], [], []
    contexts = bank["contexts"]
    for p in keep:
        pc = p["pair_class"]
        ids.append(p["pair_id"])
        cls.append(pc)
        # Constraint 2: merged pilot pairs carry BOTH ``axis`` and ``cell``
        # (same value) — ONE grouping key, the parent's cell-else-class rule.
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


# ── fire table ─────────────────────────────────────────────────────────


def load_fire(manip_path: Path) -> dict:
    """Per-(axis, value_id) fire verdicts + per-axis summary rows. Special
    axis rows (``not_in_slice`` / ``no_manipulation_check_query_class``) carry
    no floor_met — those axes are UNFILTERED (floor defaults True)."""
    doc = json.loads(Path(manip_path).read_text())
    fired: dict = {t: {} for t in FIRE_THRESHOLDS}
    rows = {}
    for r in doc.get("value_rows", []):
        key = (r["axis"], r["value_id"])
        rows[key] = r
        fired[70][key] = r["verdict"] == "fired"
        for t in (50, 90):
            fired[t][key] = r["sensitivity"][str(t)] == "fired"
    axis_rows = {r["axis"]: r for r in doc.get("axis_rows", [])}
    return {"fired": fired, "value_rows": rows, "axis_rows": axis_rows, "meta": doc.get("meta", {})}


def pair_fired_mask(pa: PairArrays, fire: dict, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """(fired_a, fired_b); values with NO fire row count FIRED (unfiltered).
    Gridless query classes (dyads + oneword) are never fire-filtered."""
    fmap = fire["fired"][threshold]

    def _ok(axis: str, vid: str) -> bool:
        return fmap.get((axis, vid), True)

    fa = np.array(
        [
            _ok(ax, va) if cl not in GRIDLESS_CLASSES else True
            for ax, va, cl in zip(pa.axis, pa.value_a, pa.cls)
        ],
        dtype=bool,
    )
    fb = np.array(
        [
            _ok(ax, vb) if cl not in GRIDLESS_CLASSES else True
            for ax, vb, cl in zip(pa.axis, pa.value_b, pa.cls)
        ],
        dtype=bool,
    )
    return fa, fb


# ── reliability (seeded split halves; ported) ──────────────────────────


def split_half_stats(st: Stores, pa: PairArrays, n_splits: int) -> dict:
    """Per-pair split-half direction reliability + noise norm at the primary
    layer's tail pooling (seeds [SPLIT_SEED, s] — parent parity)."""
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
        scores = rng.random((n_ctx, k_max))
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


# ── axis views (ported; pilot classes threaded) ────────────────────────


@dataclass
class AxisView:
    axis: str
    primary_class: str
    para_class: str | None
    primary_idx: np.ndarray
    para_idx: np.ndarray
    install_idx: np.ndarray | None
    famswap_idx: np.ndarray | None
    primary_grid: np.ndarray | None
    famswap_grid: np.ndarray | None
    primary_vps: list
    null_scheme: str
    null_kind: str  # "grid" | "pair_derangement" | "sign2"


def _grid_for(pa: PairArrays, sel: np.ndarray, n_car: int) -> tuple[np.ndarray | None, list]:
    """Complete (n_vp, n_car) grid of pair indices for a carrier-replicated
    single-carrier class; asserts completeness."""
    if sel.size == 0:
        return None, []
    vps = sorted({f"{pa.value_a[i]}-{pa.value_b[i]}" for i in sel})
    vp_of = {v: k for k, v in enumerate(vps)}
    grid = np.full((len(vps), n_car), -1, dtype=np.int64)
    for i in sel:
        grid[vp_of[f"{pa.value_a[i]}-{pa.value_b[i]}"], pa.ca[i]] = i
    assert (grid >= 0).all(), f"incomplete (vp x carrier) grid for pairs {pa.axis[sel[0]]}"
    return grid, vps


def build_axis_views(pa: PairArrays, spec: SideSpec, n_car: int) -> dict:
    idx_by: dict = {}
    for i in range(pa.n):
        idx_by.setdefault((pa.axis[i], pa.cls[i]), []).append(i)
    qpara = np.array(
        sorted(i for i in range(pa.n) if pa.cls[i] == "query_paraphrase"), dtype=np.int64
    )
    views: dict = {}
    axes_present = sorted({a for a in pa.axis if a != "query_paraphrase"})
    empty = np.array([], dtype=np.int64)
    for axis in axes_present:
        prim_cls = spec.primary_class_by_axis[axis]
        para_cls = spec.para_class_by_axis.get(axis)
        prim = np.array(sorted(idx_by.get((axis, prim_cls), [])), dtype=np.int64)
        if prim.size == 0:
            continue
        if axis in spec.query_axes:
            para = qpara
            install = None
            fams = None
            fams_grid = None
            if axis == "query_content":
                prim_grid, vps = None, [f"{pa.carrier_str[i]}" for i in prim]
                scheme = (
                    "class-preserving edge derangement over the C(12,2) carrier dyads "
                    "(carrier preservation undefined for dyadic pairs)"
                )
                kind = "pair_derangement"
            else:
                prim_grid, vps = _grid_for(pa, prim, n_car)
                scheme = "carrier- and class-preserving form-pair derangement"
                kind = "grid"
        elif axis == "query_content_oneword":
            para = empty
            install = None
            fams = None
            fams_grid = None
            prim_grid, vps = None, [f"{pa.value_a[i]}-{pa.value_b[i]}" for i in prim]
            scheme = (
                "class-preserving pair derangement over the single-carrier one-word query "
                "pairs (each value pair exists on exactly ONE carrier — grid undefined; "
                "pilot axis, plan §4.4)"
            )
            kind = "pair_derangement"
        elif axis == "answer_language":
            para = empty  # pilot: NO paraphrase family (judge has_para=False)
            install = np.array(sorted(idx_by.get((axis, "install"), [])), dtype=np.int64)
            fams = None
            fams_grid = None
            prim_grid, vps = _grid_for(pa, prim, n_car)
            scheme = (
                "carrier- and class-preserving value-pair derangement (3 language vps; "
                "pilot axis, plan §4.4)"
            )
            kind = "grid"
        else:
            para = np.array(sorted(idx_by.get((axis, para_cls), [])), dtype=np.int64)
            install = np.array(sorted(idx_by.get((axis, "install"), [])), dtype=np.int64)
            fams = np.array(sorted(idx_by.get((axis, "famswap"), [])), dtype=np.int64)
            prim_grid, vps = _grid_for(pa, prim, n_car)
            fams_grid, fams_vps = _grid_for(pa, fams, n_car) if fams.size else (None, [])
            if fams_grid is not None and prim_grid is not None:
                # famswap rows must align 1:1 with the primary grid's rows
                # (paraphrase ids f"{vid}p"; parent [g5] convention).
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
                kind = "grid"
            else:
                scheme = (
                    "sign_randomization_2value — NAMED orientation/sign-randomization null "
                    "(value-pair derangement undefined at C(2,2)=1 swap pair per carrier)"
                )
                kind = "sign2"
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
            null_kind=kind,
        )
    return views


# ── per-axis read helpers (ported) ─────────────────────────────────────


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
    if (
        view.null_kind == "grid"
        and view.primary_grid is not None
        and view.primary_grid.shape[0] >= 2
    ):
        grid = view.primary_grid
        n_vp, n_car = grid.shape
        u_obs = _unit(delta_obs[grid])
        u_pred = _unit(delta_pred[grid])
        cgrid = np.einsum("icd,jcd->ijc", u_pred, u_obs)
        perms = deranged_perms(n_vp, b_null * n_car, rng).reshape(b_null, n_car, n_vp)
        car_ix = np.arange(n_car)[None, :, None]
        vp_ix = np.arange(n_vp)[None, None, :]
        return cgrid[perms, vp_ix, car_ix].mean(axis=(1, 2))
    if view.null_kind == "pair_derangement":
        u_obs = _unit(delta_obs[view.primary_idx])
        u_pred = _unit(delta_pred[view.primary_idx])
        cmat = u_pred @ u_obs.T
        n_e = cmat.shape[0]
        if n_e < 2:
            return np.full(b_null, np.nan)
        perms = deranged_perms(n_e, b_null, rng)
        return cmat[perms, np.arange(n_e)[None, :]].mean(axis=1)
    vals = cos_sel[np.isfinite(cos_sel)]
    if vals.size == 0:
        return np.full(b_null, np.nan)
    signs = rng.integers(0, 2, size=(b_null, vals.size)) * 2 - 1
    return (signs * vals[None, :]).mean(axis=1)


def _nanmedian_quiet(a: np.ndarray, axis: int) -> np.ndarray:
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
    """Point per-vp cos of carrier means + bootstrap draws of the per-axis
    MEDIAN (ported verbatim)."""
    gb = grid_a if grid_b is None else grid_b
    da = delta_a[grid_a]
    db = delta_b[gb]
    pt = rowwise_cos(da.mean(axis=1), db.mean(axis=1))
    b_tot = mult.shape[0]
    med = np.empty(b_tot, dtype=np.float64)
    for lo in range(0, b_tot, chunk):
        m = mult[lo : lo + chunk]
        tot = np.maximum(m.sum(axis=1), 1e-12)
        ma = np.einsum("bc,vcd->bvd", m, da) / tot[:, None, None]
        mb = np.einsum("bc,vcd->bvd", m, db) / tot[:, None, None]
        med[lo : lo + chunk] = _nanmedian_quiet(rowwise_cos(ma, mb), axis=1)
    return pt, np.nanmedian(pt[None, :], axis=1), med


def pc1_identity_cos(delta_a: np.ndarray, delta_b: np.ndarray, grid: np.ndarray) -> float:
    """|cos| of top principal directions (vp-centered) of carrier-mean deltas."""
    a = delta_a[grid].mean(axis=1)
    b = delta_b[grid].mean(axis=1)
    if a.shape[0] < 2:
        return float("nan")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    va = np.linalg.svd(a, full_matrices=False)[2][0]
    vb = np.linalg.svd(b, full_matrices=False)[2][0]
    return float(abs(np.dot(va, vb)))


def offdiag_pairmean_cos(x: np.ndarray) -> float:
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
    """Bootstrap of the per-axis MEDIAN over vps of the mean pairwise
    cross-carrier cosine (duplicate carrier draws excluded; ported)."""
    u = _unit(deltas[grids])
    g = np.einsum("vcd,ved->vce", u, u)
    b_tot = idx_draws.shape[0]
    out = np.empty(b_tot, dtype=np.float64)
    for lo in range(0, b_tot, chunk):
        ix = idx_draws[lo : lo + chunk]
        gd = g[:, ix[:, :, None], ix[:, None, :]]
        distinct = ix[:, :, None] != ix[:, None, :]
        num = (gd * distinct[None]).sum(axis=(2, 3))
        den = distinct.sum(axis=(1, 2))[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = np.where(den > 0, num / np.maximum(den, 1), np.nan)
        out[lo : lo + chunk] = _nanmedian_quiet(vals, axis=0)
    return out


# ── per-side battery ───────────────────────────────────────────────────


@dataclass
class SideRun:
    spec: SideSpec
    st: Stores
    pa: PairArrays
    views: dict
    fired: dict  # threshold -> pair-level both-endpoint fired
    rel: dict
    r10: np.ndarray
    cos_arm: dict
    norm_obs: np.ndarray  # primary layer
    norm_pred: dict
    pred: dict  # arm -> (n_pairs, d) deltas
    obs_tail_primary: np.ndarray
    headline_ok: dict
    ceiling_suppressed: dict
    vp_masks: dict  # axis -> {"vp_fired": arr|None, "vp_fired_cf": arr|None}
    global_slope: dict
    global_slope_draws: dict
    slope_draws_fn: object
    wmean_fn: object
    axes_out: dict
    retrieval: dict
    perpair: list
    id_check: dict
    engine_parity: dict
    dose_ties: dict


def compute_side(
    cfg: CfgX,
    spec: SideSpec,
    bank: dict,
    st: Stores,
    fire: dict,
    mapped: dict,
    mult: np.ndarray,
    idx_draws: np.ndarray,
) -> SideRun:
    """The parent's compute_all inner battery, parameterized per SIDE.
    ``mapped``: arm -> (n_ctx, d) context-level matrices (map arm = the frozen
    map applied to vc; id arm = vc at the primary layer)."""
    t0 = time.time()
    pa = build_pair_arrays(bank, st, spec, cfg.smoke)
    n_car = len(st.carriers)
    layers = spec.store_layers

    obs_tail = {
        layer: st.va_tail_mean[layer][pa.a] - st.va_tail_mean[layer][pa.b] for layer in layers
    }
    obs_span = st.va_span_mean[spec.primary_layer][pa.a] - st.va_span_mean[spec.primary_layer][pa.b]
    delta_text = st.emb_mean[pa.a] - st.emb_mean[pa.b]
    pred = {arm: mapped[arm][pa.a] - mapped[arm][pa.b] for arm in spec.arms}
    pred_iddelta_twin = {
        layer: st.vc[layer][pa.a] - st.vc[layer][pa.b] for layer in spec.twin_layers
    }

    id_check = identity_cancellation_check(
        st.vc[spec.primary_layer], pa.a, pa.b, np.random.default_rng([BOOT_SEED, 999])
    )
    engine_parity = assert_engine_parity(spec.name, st.emb_engine, cfg.embed_parity_report)

    cos_arm = {arm: rowwise_cos(pred[arm], obs_tail[spec.primary_layer]) for arm in spec.arms}
    cos_arm_span = {arm: rowwise_cos(pred[arm], obs_span) for arm in spec.arms}
    norm_obs = {layer: np.linalg.norm(obs_tail[layer], axis=1) for layer in layers}
    norm_obs_span = np.linalg.norm(obs_span, axis=1)
    norm_pred = {arm: np.linalg.norm(pred[arm], axis=1) for arm in spec.arms}
    norm_text = np.linalg.norm(delta_text, axis=1)
    dlen = st.ans_len_mean[pa.a] - st.ans_len_mean[pa.b]

    rel = split_half_stats(st, pa, cfg.n_splits)
    r10 = rel["r_full"]

    fired = {}
    fa_fb = {}
    for t in FIRE_THRESHOLDS:
        fa, fb = pair_fired_mask(pa, fire, t)
        fa_fb[t] = (fa, fb)
        fired[t] = fa & fb

    # edit-dose pooled OLS + residuals + tie report (convention 19)
    dose = pa.changed.astype(np.float64)
    dose_fit = {}
    resid = {}
    for name, norms in {"observed": norm_obs[spec.primary_layer], **norm_pred}.items():
        icpt, slope = ols_intercept_slope(norms, dose)
        dose_fit[name] = {"intercept": icpt, "slope": slope, "n": int(pa.n)}
        resid[name] = norms - (icpt + slope * dose)

    views = build_axis_views(pa, spec, n_car)
    if not cfg.smoke:
        expected_axes = spec.instruction_axes + spec.query_axes + spec.pilot_axes
        missing_axes = [a for a in expected_axes if a not in views]
        assert not missing_axes, f"axes missing from production stores: {missing_axes}"

    def wm(vals: np.ndarray, sel: np.ndarray) -> tuple[float, list]:
        pt = float(np.nanmean(vals[sel])) if sel.size else float("nan")
        if sel.size == 0:
            return pt, [float("nan"), float("nan")]
        draws = boot_weighted_mean(vals[sel], pa.ca[sel], pa.cb[sel], pa.dyad[sel], mult)
        return pt, _ci(draws)

    def _nm(vals: np.ndarray, sel: np.ndarray) -> float:
        if sel.size == 0:
            return float("nan")
        v = vals[sel]
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    def slope_draws(sel: np.ndarray, arm: str, m: np.ndarray | None = None) -> np.ndarray:
        mm = mult if m is None else m
        num = boot_pair_sums(
            norm_pred[arm][sel] * norm_obs[spec.primary_layer][sel],
            pa.ca[sel],
            pa.cb[sel],
            pa.dyad[sel],
            mm,
        )
        den = boot_pair_sums(
            norm_obs[spec.primary_layer][sel] ** 2, pa.ca[sel], pa.cb[sel], pa.dyad[sel], mm
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    def wmean_draws(vals: np.ndarray, sel: np.ndarray, m: np.ndarray | None = None) -> np.ndarray:
        mm = mult if m is None else m
        if sel.size == 0:
            return np.full(mm.shape[0], np.nan)
        return boot_weighted_mean(vals[sel], pa.ca[sel], pa.cb[sel], pa.dyad[sel], mm)

    all_idx = np.arange(pa.n)
    swap_idx = np.array([i for i in all_idx if pa.cls[i] == "swap"], dtype=np.int64)
    global_slope = {
        arm: through_origin_slope(norm_pred[arm], norm_obs[spec.primary_layer]) for arm in spec.arms
    }
    global_slope_swap = {
        arm: through_origin_slope(norm_pred[arm][swap_idx], norm_obs[spec.primary_layer][swap_idx])
        for arm in spec.arms
    }
    global_slope_draws = {arm: slope_draws(all_idx, arm) for arm in spec.arms}
    global_slope_swap_draws = {arm: slope_draws(swap_idx, arm) for arm in spec.arms}

    axes_out: dict = {}
    null_schemes: dict = {}
    headline_ok_by: dict = {}
    suppressed_by: dict = {}
    vp_masks: dict = {}
    dose_ties: dict = {}
    side_off = NULL_OFFSET[spec.name]
    for k, (axis, view) in enumerate(sorted(views.items())):
        ta = time.time()
        prim = view.primary_idx
        hmask = fired[70][prim]
        ar = fire["axis_rows"].get(axis)
        # special rows (not_in_slice / no_manipulation_check_query_class) carry
        # no floor_met -> unfiltered (the parent's absent-axis convention).
        floor_met = bool(ar.get("floor_met", True)) if ar is not None else True
        compliance_limited = ar is not None and "floor_met" in ar and not floor_met
        no_fired_pairs = not bool(hmask.any())
        headline_ok = not compliance_limited and not no_fired_pairs
        head = prim[hmask] if headline_ok else np.array([], dtype=np.int64)
        null_schemes[axis] = view.null_scheme
        headline_ok_by[axis] = headline_ok

        uniq, counts_d = np.unique(pa.changed[prim], return_counts=True)
        dose_ties[axis] = {
            "n_distinct_dose_values": int(uniq.size),
            "modal_dose_fraction": float(counts_d.max() / max(prim.size, 1)),
        }

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

        ceil_pt, ceil_ci = wm(r10, head)
        ceil_all_pt, ceil_all_ci = wm(r10, prim)
        rel_axis = {
            "r_half_mean": _nm(rel["r_half"], head),
            "r10_mean": ceil_pt,
            "r10_ci95": ceil_ci,
            "r10_mean_all_values": ceil_all_pt,
            "r10_ci95_all_values": ceil_all_ci,
            "noise_norm_mean": _nm(rel["noise_norm"], head),
            "noise_norm_mean_all_values": _nm(rel["noise_norm"], prim),
            "spearman_brown": "r10 = 2*r5 / (1 + r5)",
        }
        suppressed = suppression_verdict(ceil_pt, ceil_ci[0], ceil_ci[1])
        suppressed_by[axis] = suppressed

        rng_null = np.random.default_rng([NULL_SEED, side_off + k])
        direction = {}
        for arm in spec.arms:
            pt, ci = wm(cos_arm[arm], head)
            pt_all, ci_all = wm(cos_arm[arm], prim)
            nd = direction_null_draws(
                view,
                obs_tail[spec.primary_layer],
                pred[arm],
                cos_arm[arm][prim],
                cfg.b_null,
                rng_null,
            )
            ratio = float("nan") if suppressed else pt / ceil_pt
            controls = {}
            para_ctl_name = (
                "query_paraphrase" if axis in spec.query_axes else "instruction_paraphrase"
            )
            for cname, cidx in {
                "install": view.install_idx,
                para_ctl_name: view.para_idx if view.para_idx.size else None,
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
                    "seed": [NULL_SEED, side_off + k],
                    "over": "all primary-class value pairs (fire mask NOT applied to the null)",
                },
                "ceiling_normalized_cos": None if suppressed else ratio,
                "ceiling_suppressed": suppressed,
                "controls": controls,
            }
            if arm != spec.id_arm:
                gpt, gci = wm(cos_arm[arm] - cos_arm[spec.id_arm], head)
                gpt_all, gci_all = wm(cos_arm[arm] - cos_arm[spec.id_arm], prim)
                direction[arm]["gap_vs_iddelta"] = {
                    "mean_cos_gap_headline": gpt,
                    "ci95": gci,
                    "mean_cos_gap_all_values": gpt_all,
                    "ci95_all_values": gci_all,
                    "paired": "per-pair cos difference under the SHARED carrier bootstrap",
                }

        calibration = {}
        for arm in spec.arms:
            ax_pt = through_origin_slope(norm_pred[arm][head], norm_obs[spec.primary_layer][head])
            ax_all_pt = through_origin_slope(
                norm_pred[arm][prim], norm_obs[spec.primary_layer][prim]
            )
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
                "global_slope_all_pairs": global_slope[arm],
                "ratio_to_global": ax_pt / global_slope[arm] if global_slope[arm] else float("nan"),
                "ratio_to_global_ci95": _ci(ratio_draws),
                "ratio_to_global_all_values": (
                    ax_all_pt / global_slope[arm] if global_slope[arm] else float("nan")
                ),
                "ratio_to_global_ci95_all_values": _ci(ratio_all_draws),
                "global_slope_swap": global_slope_swap[arm],
                "n_swap_pairs_global": int(swap_idx.size),
                "ratio_to_global_swap": (
                    ax_pt / global_slope_swap[arm] if global_slope_swap[arm] else float("nan")
                ),
                "ratio_to_global_swap_ci95": _ci(ratio_swap_draws),
                "ratio_to_global_swap_all_values": (
                    ax_all_pt / global_slope_swap[arm] if global_slope_swap[arm] else float("nan")
                ),
                "ratio_to_global_swap_ci95_all_values": _ci(ratio_swap_all_draws),
            }

        identity: dict = {}
        vp_fired = None
        if view.primary_grid is not None:
            vp_fired = fired[70][view.primary_grid[:, 0]]
            grid_head = view.primary_grid[vp_fired] if (headline_ok and vp_fired.any()) else None
            med_head_draws: dict = {}
            med_all_draws: dict = {}
            for arm in spec.arms:
                pt_rows, _, med_draws = carrier_mean_cos_median(
                    view.primary_grid, None, obs_tail[spec.primary_layer], pred[arm], mult
                )
                med_all_draws[arm] = med_draws
                if grid_head is not None:
                    pt_rows_h, _, med_draws_h = carrier_mean_cos_median(
                        grid_head, None, obs_tail[spec.primary_layer], pred[arm], mult
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
                        obs_tail[spec.primary_layer], pred[arm], view.primary_grid
                    ),
                }
            for arm in spec.arms:
                if arm == spec.id_arm:
                    continue
                da_, di_ = med_head_draws[arm], med_head_draws[spec.id_arm]
                if da_ is not None and di_ is not None:
                    identity[arm]["median_gap_vs_iddelta"] = {
                        "gap": identity[arm]["median"] - identity[spec.id_arm]["median"],
                        "ci95": _ci(da_ - di_),
                        "paired": "median-draw difference under the SHARED carrier bootstrap",
                    }
                else:
                    identity[arm]["median_gap_vs_iddelta"] = None
                identity[arm]["median_gap_vs_iddelta_all_values"] = {
                    "gap": (
                        identity[arm]["median_all_values"]
                        - identity[spec.id_arm]["median_all_values"]
                    ),
                    "ci95": _ci(med_all_draws[arm] - med_all_draws[spec.id_arm]),
                    "paired": "median-draw difference under the SHARED carrier bootstrap",
                }
        else:
            identity = {
                "n/a": "no carrier-replicated value pair exists for this class — the "
                "carrier-mean identity read is undefined (dyads / single-carrier pilot pairs)"
            }

        cross_family: dict = {}
        vp_fired_cf = None
        if view.famswap_grid is not None and view.primary_grid is not None:
            rng_cf = np.random.default_rng([NULL_SEED, side_off + 500 + k])
            vp_fired_cf = fired[70][view.primary_grid[:, 0]] & fired[70][view.famswap_grid[:, 0]]
            cf_head_ok = headline_ok and bool(vp_fired_cf.any())
            spaces = {"observed": (obs_tail[spec.primary_layer], obs_tail[spec.primary_layer])}
            for arm in spec.arms:
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
                        "over": "all primary-class value pairs (fire mask NOT applied)",
                    },
                }
        else:
            cross_family = {"n/a": "no paraphrase-family swap class for this axis"}
        vp_masks[axis] = {"vp_fired": vp_fired, "vp_fired_cf": vp_fired_cf}

        para_head = (
            view.para_idx[fired[70][view.para_idx]]
            if headline_ok and view.para_idx.size
            else np.array([], dtype=np.int64)
        )
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
            "no_para_family": bool(view.para_idx.size == 0),
            "note": "Qwen3-Embedding-8B mean answer embeddings (means of L2-normalized "
            "per-draw rows, NOT re-normalized); observed only",
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

        surface = {}
        for name, norms in {"observed": norm_obs[spec.primary_layer], **norm_pred}.items():
            fpt, fci = wm(norms, head)
            ppt, pci = wm(norms, para_head)
            fpt_all, fci_all = wm(norms, prim)
            ppt_all, pci_all = wm(norms, view.para_idx)
            rf, _ = wm(resid[name], head)
            rp, _ = wm(resid[name], para_head)
            rf_all, _ = wm(resid[name], prim)
            rp_all, _ = wm(resid[name], view.para_idx)
            if head.size and para_head.size:
                gap_ci = _ci(wmean_draws(norms, head) - wmean_draws(norms, para_head))
            else:
                gap_ci = [float("nan"), float("nan")]
            if prim.size and view.para_idx.size:
                gap_all_ci = _ci(wmean_draws(norms, prim) - wmean_draws(norms, view.para_idx))
            else:
                gap_all_ci = [float("nan"), float("nan")]
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
                "gap_ci95_all_values": gap_all_ci,
                "edit_dose_ols": dose_fit[name],
                "edit_dose_ties": dose_ties[axis],
                "residualized_gap": rf - rp,
                "residualized_gap_all_values": rf_all - rp_all,
                "labeling": "DESCRIPTIVE only (plan §6: no H-label keys on this gap)",
            }

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

        layer_twins: dict = {}
        for layer in spec.twin_layers:
            c = rowwise_cos(pred_iddelta_twin[layer], obs_tail[layer])
            no_l = np.linalg.norm(pred_iddelta_twin[layer], axis=1)
            ax_slope = through_origin_slope(no_l[head], norm_obs[layer][head])
            ax_slope_all = through_origin_slope(no_l[prim], norm_obs[layer][prim])
            gl = through_origin_slope(no_l, norm_obs[layer])
            layer_twins[str(layer)] = {
                "iddelta_mean_cos_headline": _nm(c, head),
                "iddelta_mean_cos_all_values": _nm(c, prim),
                "iddelta_ratio_to_global": ax_slope / gl if gl else float("nan"),
                "iddelta_ratio_to_global_all_values": ax_slope_all / gl if gl else float("nan"),
                "note": "iddelta only — the frozen map is primary-layer-fit; SENSITIVITY "
                "twin, NEVER a cross-model read (cross-unit constraint 5)",
            }
        if not spec.twin_layers:
            layer_twins = {
                "n/a": "no twin layers on this side (constraint 5: the banked 7B extras "
                "have no 9B counterpart; twins never enter a cross-model contrast)"
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
            for arm in spec.arms
        }

        axes_out[axis] = {
            "axis": axis,
            "model_tag": spec.name,
            "pilot_axis": axis in spec.pilot_axes,
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
        }
        if axis in spec.pilot_axes:
            axes_out[axis]["cross_model_status"] = PILOT_LABEL
        print(
            f"[an:{spec.name}] axis {k + 1}/{len(views)} {axis} elapsed={time.time() - ta:.1f}s",
            flush=True,
        )

    retrieval: dict = {"global": {}, "per_axis": {}}
    pool = obs_tail[spec.primary_layer]
    for arm in spec.arms:
        retrieval["global"][arm] = {
            metric: knn_retrieval(pred[arm], pool, ks=(1, 5, 10), metric=metric, pool=pool)
            for metric in ("cosine", "euclidean")
        }
    retrieval["chance"] = {"rule": "chance = k / n_pool", "n_pool_global": int(pa.n)}
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
            for arm in spec.arms
        }
        retrieval["per_axis"][axis]["n_pool"] = int(sel.size)

    fa70, fb70 = fa_fb[70]
    perpair: list = []
    for i in range(pa.n):
        perpair.append(
            {
                "model_tag": spec.name,
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
                "norm_obs_tail_primary": float(norm_obs[spec.primary_layer][i]),
                "norm_obs_span_primary": float(norm_obs_span[i]),
                "norm_text": float(norm_text[i]),
                "cos": {arm: float(cos_arm[arm][i]) for arm in spec.arms},
                "cos_span": {arm: float(cos_arm_span[arm][i]) for arm in spec.arms},
                "norm_pred": {arm: float(norm_pred[arm][i]) for arm in spec.arms},
                "r_half": float(rel["r_half"][i]),
                "r10": float(r10[i]),
                "noise_norm": float(rel["noise_norm"][i]),
                "fired_a_70": bool(fa70[i]),
                "fired_b_70": bool(fb70[i]),
                "pair_fired_70": bool(fa70[i] and fb70[i]),
                "in_headline_70": bool(
                    fa70[i] and fb70[i] and headline_ok_by.get(pa.axis[i], True)
                ),
                "pilot_axis": pa.axis[i] in spec.pilot_axes,
            }
        )

    print(f"[an:{spec.name}] battery done in {time.time() - t0:.1f}s", flush=True)
    return SideRun(
        spec=spec,
        st=st,
        pa=pa,
        views=views,
        fired=fired,
        rel=rel,
        r10=r10,
        cos_arm=cos_arm,
        norm_obs=norm_obs[spec.primary_layer],
        norm_pred=norm_pred,
        pred=pred,
        obs_tail_primary=obs_tail[spec.primary_layer],
        headline_ok=headline_ok_by,
        ceiling_suppressed=suppressed_by,
        vp_masks=vp_masks,
        global_slope=global_slope,
        global_slope_draws=global_slope_draws,
        slope_draws_fn=slope_draws,
        wmean_fn=wmean_draws,
        axes_out=axes_out,
        retrieval=retrieval,
        perpair=perpair,
        id_check=id_check,
        engine_parity=engine_parity,
        dose_ties=dose_ties,
    )


def pilot_placement_block(run: SideRun) -> dict:
    """Convention 18: rank ALL of this side's axes by ONE common statistic —
    obs SNR = mean observed flip norm / mean split-half noise norm over the
    primary pairs (all-values; para-free pilots stay comparable) — and report
    each pilot axis's rank + quartile."""
    snr: dict = {}
    for axis, view in run.views.items():
        prim = view.primary_idx
        flip = float(np.nanmean(run.norm_obs[prim])) if prim.size else float("nan")
        noise = (
            float(np.nanmean(run.rel["noise_norm"][prim]))
            if prim.size and np.isfinite(run.rel["noise_norm"][prim]).any()
            else float("nan")
        )
        snr[axis] = flip / noise if noise and np.isfinite(noise) and noise > 0 else float("nan")
    finite = {a: v for a, v in snr.items() if np.isfinite(v)}
    ranked = sorted(finite, key=lambda a: -finite[a])
    out = {
        "statistic": "obs_snr = mean(||obs delta||) / mean(split-half noise norm), primary "
        "pairs (all-values) — ONE common statistic across all axes (convention 18)",
        "snr_by_axis": snr,
        "rank_order_desc": ranked,
        "pilots": {},
    }
    n = len(ranked)
    for axis in run.spec.pilot_axes:
        if axis in ranked and n:
            r = ranked.index(axis) + 1
            out["pilots"][axis] = {"rank": r, "of": n, "quartile": int(np.ceil(4.0 * r / n))}
        else:
            out["pilots"][axis] = {"rank": None, "of": n, "quartile": None}
    return out


# ── H1 (frozen layer pair; paired test-row bootstrap) ──────────────────


def _pooled_r2(pred: np.ndarray, target: np.ndarray) -> float:
    """Whole-map variance-weighted R2 = 1 - sum SSE / sum SST (the
    issue1491_ladder_fits/#2564 parity arithmetic)."""
    sse = float(((target - pred) ** 2).sum())
    sst = float(((target - target.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - sse / (sst + 1e-30)


def _pooled_r2_draws(pred: np.ndarray, target: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Vectorized pooled R2 per bootstrap draw. counts: (B, n) row
    multiplicities (each row sums to n). SSE = counts @ per-row SSE; SST is
    recomputed about each draw's OWN resampled mean."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    n = target.shape[0]
    sse_row = ((target - pred) ** 2).sum(axis=1)
    t2_row = (target**2).sum(axis=1)
    sse = counts @ sse_row
    s1 = counts @ target  # (B, d) resampled sums
    n_tot = counts.sum(axis=1)
    assert np.allclose(n_tot, n), "row-bootstrap counts must sum to n per draw"
    sst = counts @ t2_row - (s1**2).sum(axis=1) / n_tot
    return 1.0 - sse / (sst + 1e-30)


def h1_verdict(lo: float, hi: float) -> str:
    """Plan §3 H1 lattice — DISJOINT and EXHAUSTIVE over finite CIs:
    CI excludes 0 negative -> consistent; excludes 0 positive -> contradicted;
    includes 0 -> inconclusive (a LIVE first-class verdict, never an error)."""
    assert np.isfinite(lo) and np.isfinite(hi) and lo <= hi, (lo, hi)
    if hi < 0.0:
        return "h1_consistent"
    if lo > 0.0:
        return "h1_contradicted"
    return "h1_inconclusive"


def compute_h1(
    preds9: dict, preds7: dict, lstar: int, b_boot: int, rng: np.random.Generator
) -> tuple[dict, dict]:
    """Delta_map = R2_9B(L*) - R2_7B(L19) on the realized shared test rows,
    layer pair FROZEN; paired TEST-ROW bootstrap (ONE shared row-resample per
    draw); per-draw deltas returned for persistence."""
    assert int(preds9["layer"]) == int(lstar), (preds9["layer"], lstar)
    ci9 = [str(x) for x in preds9["ci_te"]]
    ci7 = [str(x) for x in preds7["ci_te"]]
    # constraint: the REALIZED SHARED ROW INTERSECTION must be the full split,
    # asserted as EXACT ORDERED id equality (fail loud on any drift).
    assert ci9 == ci7, (
        "H1 test-row id mismatch (ordered): "
        f"n9={len(ci9)} n7={len(ci7)} first_diff="
        f"{next((i for i, (a, b) in enumerate(zip(ci9, ci7)) if a != b), 'length')}"
    )
    p9 = np.asarray(preds9["pred_te"], dtype=np.float64)
    t9 = np.asarray(preds9["target_te"], dtype=np.float64)
    p7 = np.asarray(preds7["pred_te"], dtype=np.float64)
    t7 = np.asarray(preds7["target_te"], dtype=np.float64)
    n = len(ci9)
    assert p9.shape == t9.shape and p9.shape[0] == n, (p9.shape, t9.shape, n)
    assert p7.shape == t7.shape and p7.shape[0] == n, (p7.shape, t7.shape, n)
    r2_9 = _pooled_r2(p9, t9)
    r2_7 = _pooled_r2(p7, t7)
    delta_pt = r2_9 - r2_7
    idx = rng.integers(0, n, size=(b_boot, n))
    counts = np.zeros((b_boot, n), dtype=np.float64)
    np.add.at(counts, (np.repeat(np.arange(b_boot), n), idx.ravel()), 1.0)
    r9_draws = _pooled_r2_draws(p9, t9, counts)
    r7_draws = _pooled_r2_draws(p7, t7, counts)
    delta_draws = r9_draws - r7_draws
    lo, hi = _ci(delta_draws)
    doc = {
        "definition": "Delta_map = R2_9B(L*) - R2_7B(L19), pooled R2 on the realized shared "
        "test-row intersection; layer pair FROZEN (L* read from unit 4's freeze)",
        "layer_pair": {"qwen35_9b": int(lstar), "qwen25_7b": L19},
        "n_test_rows": n,
        "r2_9b_lstar": r2_9,
        "r2_7b_l19": r2_7,
        "delta_map": delta_pt,
        "delta_ci95": [lo, hi],
        "bootstrap": {
            "scheme": "paired TEST-ROW bootstrap — ONE shared row resample per draw, both "
            "sides evaluated on it; pooled R2 recomputed about each draw's own mean",
            "B": b_boot,
            "seed": list(H1_BOOT_SEED),
        },
        "verdict": h1_verdict(lo, hi),
        "verdict_rule": "hi < 0 -> h1_consistent; lo > 0 -> h1_contradicted; else "
        "h1_inconclusive (disjoint + exhaustive; plan §3)",
        "narration": "inconclusive is a LIVE first-class verdict (convention 14): the plan "
        "expectation range [0.68, 0.73] straddles the 7B anchor 0.7250873",
    }
    # OPTIONAL wc companion (same machinery; NON-verdict-bearing).
    if all(k in preds9 for k in ("ci_wc", "pred_wc")) and all(
        k in preds7 for k in ("ci_wc", "pred_wc")
    ):
        cw9 = [str(x) for x in preds9["ci_wc"]]
        cw7 = [str(x) for x in preds7["ci_wc"]]
        if cw9 == cw7:
            pw9 = np.asarray(preds9["pred_wc"], dtype=np.float64)
            tw9 = np.asarray(preds9["target_wc"], dtype=np.float64)
            pw7 = np.asarray(preds7["pred_wc"], dtype=np.float64)
            tw7 = np.asarray(preds7["target_wc"], dtype=np.float64)
            doc["companion_wc"] = {
                "n_rows": len(cw9),
                "r2_9b": _pooled_r2(pw9, tw9),
                "r2_7b": _pooled_r2(pw7, tw7),
                "delta": _pooled_r2(pw9, tw9) - _pooled_r2(pw7, tw7),
                "note": "wildchat companion — descriptive, NOT verdict-bearing",
            }
        else:
            doc["companion_wc"] = {"skipped": "wc row ids differ across sides"}
    return doc, {"delta_draws": delta_draws, "r9_draws": r9_draws, "r7_draws": r7_draws}


# ── cross-model contrasts (§4.6) + H2 ──────────────────────────────────


def assert_frozen_layer_pair(layer_9b: int, layer_7b: int, lstar: int) -> None:
    """Cross-unit constraint 5: cross-model reads use EXACTLY the frozen
    layer pair (L*, L19); twin layers never enter a cross-model contrast."""
    if layer_7b != L19:
        raise RuntimeError(f"cross-model 7B layer must be L19, got {layer_7b}")
    if layer_9b != lstar:
        raise RuntimeError(
            f"cross-model 9B layer must be the frozen L*={lstar}, got {layer_9b} — "
            "the {16,22,30} twins have NO 7B counterpart (constraint 5)"
        )


def resolve_primary_h2_arm(candidate_7b_arms: list) -> str:
    """Assert the UNIQUE pinned H2 primary 7B map arm (plan §4.5)."""
    primaries = [a for a in candidate_7b_arms if a == PRIMARY_H2_7B_ARM]
    if len(primaries) != 1:
        raise RuntimeError(
            f"primary_h2_7b_arm must be uniquely {PRIMARY_H2_7B_ARM!r}; got {candidate_7b_arms}"
        )
    if REF_7B_PARENT in candidate_7b_arms:
        raise RuntimeError(f"{REF_7B_PARENT} is a sensitivity read — NEVER a primary H2 arm")
    return PRIMARY_H2_7B_ARM


def _ref7b_stat(ref_axes: dict, axis: str, stat: str) -> float:
    """Point extraction of a scale-free statistic from the parent's committed
    minpair_delta.json (arm_779ce = the parent's primary frozen map)."""

    def _f(v) -> float:
        return float(v) if v is not None and np.isfinite(float(v)) else float("nan")

    ax = ref_axes.get(axis)
    if ax is None:
        return float("nan")
    try:
        if stat == "direction_cos":
            return _f(ax["direction"]["arm_779ce"]["mean_cos_headline"])
        if stat == "calibration_ratio_to_global":
            return _f(ax["calibration"]["arm_779ce"]["ratio_to_global"])
        if stat == "crossfam_cos_observed":
            return _f(ax["cross_family"]["observed"]["median"])
        if stat == "crossfam_cos_maparm":
            return _f(ax["cross_family"]["arm_779ce"]["median"])
        if stat == "obs_separation_snr":
            flip = ax["surface"]["observed"]["flip_norm_mean"]
            noise = ax["reliability"]["noise_norm_mean"]
            if flip is None or noise is None or not noise:
                return float("nan")
            return _f(float(flip) / float(noise))
        if stat == "axis_identity_cos":
            return _f(ax["identity"]["arm_779ce"]["median"])
    except (KeyError, TypeError):
        return float("nan")
    return float("nan")


def crossmodel_contrasts(
    run9: SideRun,
    run7: SideRun,
    lstar: int,
    mult: np.ndarray,
    ref7b_doc: dict,
    ref7b_commit: str,
    cfg: CfgX,
) -> tuple[dict, dict]:
    """§4.6: per axis x scale-free statistic — s_7B and s_9B side by side +
    carrier-paired cross-model delta under ONE shared carrier resample, with
    t_11 + LOCO companions (convention 15) and Spearman blocks (raw + exact
    permutation p + changed_tokens partial + ceiling-cleared companion)."""
    assert_frozen_layer_pair(run9.spec.primary_layer, run7.spec.primary_layer, lstar)
    assert run9.st.carriers == run7.st.carriers, (run9.st.carriers, run7.st.carriers)
    n_car = len(run9.st.carriers)
    loco = loco_multiplicities(n_car)
    parent_axes = tuple(sorted(run7.views.keys()))
    assert len(parent_axes) == len(run7.spec.instruction_axes) + len(run7.spec.query_axes) or (
        cfg.smoke
    ), parent_axes

    # pair_id-aligned shared subset (identity fields asserted equal).
    id9 = {pid: i for i, pid in enumerate(run9.pa.ids)}
    shared: dict = {}
    for axis in parent_axes:
        v7 = run7.views[axis]
        i7 = v7.primary_idx
        i9 = np.array([id9[run7.pa.ids[j]] for j in i7], dtype=np.int64)
        for j7, j9 in zip(i7[: min(len(i7), 5)], i9[:5]):
            assert run7.pa.cls[j7] == run9.pa.cls[j9], (axis, run7.pa.cls[j7], run9.pa.cls[j9])
        assert [run7.pa.value_a[j] for j in i7] == [run9.pa.value_a[j] for j in i9], axis
        assert [run7.pa.value_b[j] for j in i7] == [run9.pa.value_b[j] for j in i9], axis
        assert (run7.pa.ca[i7] == run9.pa.ca[i9]).all(), axis
        sym = run9.fired[70][i9] & run7.fired[70][i7]
        head_ok = bool(run9.headline_ok.get(axis, True)) and bool(run7.headline_ok.get(axis, True))
        use = (i9[sym], i7[sym]) if (head_ok and sym.any()) else (i9, i7)
        shared[axis] = {
            "i9": use[0],
            "i7": use[1],
            "i9_all": i9,
            "i7_all": i7,
            "symmetric_headline": bool(head_ok and sym.any()),
            "n_shared_primary": int(len(i7)),
            "n_symmetric_fired": int(sym.sum()),
            "n_dropped_9b_only": int((run7.fired[70][i7] & ~run9.fired[70][i9]).sum()),
            "n_dropped_7b_only": int((run9.fired[70][i9] & ~run7.fired[70][i7]).sum()),
        }

    # changed_tokens per axis, per tokenizer (convention 16 covariates)
    ct9 = np.array([float(np.mean(run9.pa.changed[shared[a]["i9_all"]])) for a in parent_axes])
    ct7 = np.array([float(np.mean(run7.pa.changed[shared[a]["i7_all"]])) for a in parent_axes])

    def _wmean(run: SideRun, vals: np.ndarray, sel: np.ndarray) -> float:
        return float(np.nanmean(vals[sel])) if sel.size else float("nan")

    def _wdraws(run: SideRun, vals: np.ndarray, sel: np.ndarray, m: np.ndarray) -> np.ndarray:
        if sel.size == 0:
            return np.full(m.shape[0], np.nan)
        return boot_weighted_mean(vals[sel], run.pa.ca[sel], run.pa.cb[sel], run.pa.dyad[sel], m)

    def stat_direction(run: SideRun, axis: str, key9: str, m: np.ndarray):
        sel = shared[axis][key9]
        vals = run.cos_arm[run.spec.map_arm]
        return _wmean(run, vals, sel), _wdraws(run, vals, sel, m)

    def stat_calibration(run: SideRun, axis: str, key9: str, m: np.ndarray):
        sel = shared[axis][key9]
        ax_pt = through_origin_slope(run.norm_pred[run.spec.map_arm][sel], run.norm_obs[sel])
        gl = run.global_slope[run.spec.map_arm]
        pt = ax_pt / gl if gl else float("nan")
        ax_draws = run.slope_draws_fn(sel, run.spec.map_arm, m)
        gl_draws = run.slope_draws_fn(np.arange(run.pa.n), run.spec.map_arm, m)
        with np.errstate(invalid="ignore", divide="ignore"):
            return pt, ax_draws / gl_draws

    def stat_separation(run: SideRun, axis: str, key: str, m: np.ndarray):
        selp = shared[axis][key]
        flip = _wmean(run, run.norm_obs, selp)
        noise = _wmean(run, run.rel["noise_norm"], selp)
        pt = flip / noise if noise and np.isfinite(noise) and noise > 0 else float("nan")
        fd = _wdraws(run, run.norm_obs, selp, m)
        nd = _wdraws(run, run.rel["noise_norm"], selp, m)
        with np.errstate(invalid="ignore", divide="ignore"):
            return pt, np.where(nd > 0, fd / nd, np.nan)

    def _sym_grid(run_a: SideRun, run_b: SideRun, axis: str, cf: bool):
        va, vb = run_a.views[axis], run_b.views[axis]
        if va.primary_grid is None or vb.primary_grid is None:
            return None
        assert va.primary_vps == vb.primary_vps, (axis, va.primary_vps, vb.primary_vps)
        key = "vp_fired_cf" if cf else "vp_fired"
        ma = run_a.vp_masks[axis][key]
        mb = run_b.vp_masks[axis][key]
        if cf and (va.famswap_grid is None or vb.famswap_grid is None):
            return None
        if ma is None or mb is None:
            mask = np.ones(va.primary_grid.shape[0], dtype=bool)
        else:
            mask = ma & mb
        if not mask.any():
            mask = np.ones(va.primary_grid.shape[0], dtype=bool)
        return mask

    def stat_crossfam(run: SideRun, axis: str, space_pred: bool, mask, m: np.ndarray):
        v = run.views[axis]
        if v.famswap_grid is None or v.primary_grid is None or mask is None:
            return float("nan"), np.full(m.shape[0], np.nan)
        da = run.pred[run.spec.map_arm] if space_pred else run.obs_tail_primary
        pt_rows, _, med = carrier_mean_cos_median(
            v.primary_grid[mask], v.famswap_grid[mask], da, da, m
        )
        return float(np.nanmedian(pt_rows)), med

    def stat_identity(run: SideRun, axis: str, mask, m: np.ndarray):
        v = run.views[axis]
        if v.primary_grid is None or mask is None:
            return float("nan"), np.full(m.shape[0], np.nan)
        pt_rows, _, med = carrier_mean_cos_median(
            v.primary_grid[mask], None, run.obs_tail_primary, run.pred[run.spec.map_arm], m
        )
        return float(np.nanmedian(pt_rows)), med

    stats_def = {
        "direction_cos": "map-arm mean direction cos over symmetric-fired shared primary pairs",
        "calibration_ratio_to_global": "map-arm axis/global through-origin norm-slope ratio",
        "crossfam_cos_observed": "observed-space cross-family consistency median "
        "(instruction axes only)",
        "crossfam_cos_maparm": "map-arm predicted-space cross-family consistency median "
        "(instruction axes only)",
        "obs_separation_snr": "observed-space separation, ceiling-adjusted: mean ||obs delta|| "
        "/ mean split-half noise norm (symmetric-fired shared primary pairs)",
        "axis_identity_cos": "map-arm carrier-mean axis-identity median (grid axes only)",
    }

    perdraw: dict = {}
    stat_tables: dict = {}
    rng_mc = np.random.default_rng([NULL_SEED, 9999])
    for stat in stats_def:
        rows = []
        d9_all, d7_all, dd_all, loco_all, axes_used = [], [], [], [], []
        for axis in parent_axes:
            cf = stat.startswith("crossfam")
            grid_stat = cf or stat == "axis_identity_cos"
            mask = _sym_grid(run9, run7, axis, cf) if grid_stat else None
            if stat == "direction_cos":
                p9, dr9 = stat_direction(run9, axis, "i9", mult)
                p7, dr7 = stat_direction(run7, axis, "i7", mult)
                l9 = stat_direction(run9, axis, "i9", loco)[1]
                l7 = stat_direction(run7, axis, "i7", loco)[1]
            elif stat == "calibration_ratio_to_global":
                p9, dr9 = stat_calibration(run9, axis, "i9", mult)
                p7, dr7 = stat_calibration(run7, axis, "i7", mult)
                l9 = stat_calibration(run9, axis, "i9", loco)[1]
                l7 = stat_calibration(run7, axis, "i7", loco)[1]
            elif stat == "obs_separation_snr":
                p9, dr9 = stat_separation(run9, axis, "i9", mult)
                p7, dr7 = stat_separation(run7, axis, "i7", mult)
                l9 = stat_separation(run9, axis, "i9", loco)[1]
                l7 = stat_separation(run7, axis, "i7", loco)[1]
            elif cf:
                space_pred = stat == "crossfam_cos_maparm"
                if mask is None:
                    continue
                p9, dr9 = stat_crossfam(run9, axis, space_pred, mask, mult)
                p7, dr7 = stat_crossfam(run7, axis, space_pred, mask, mult)
                l9 = stat_crossfam(run9, axis, space_pred, mask, loco)[1]
                l7 = stat_crossfam(run7, axis, space_pred, mask, loco)[1]
            else:  # axis_identity_cos
                if mask is None:
                    continue
                p9, dr9 = stat_identity(run9, axis, mask, mult)
                p7, dr7 = stat_identity(run7, axis, mask, mult)
                l9 = stat_identity(run9, axis, mask, loco)[1]
                l7 = stat_identity(run7, axis, mask, loco)[1]
            delta_pt = p9 - p7
            dd = dr9 - dr7
            ld = l9 - l7
            sd = float(np.nanstd(dd, ddof=1)) if np.isfinite(dd).sum() >= 2 else float("nan")
            rows.append(
                {
                    "axis": axis,
                    "s_9b": p9,
                    "s_7b": p7,
                    "s_7b_ref_parent": _ref7b_stat(ref7b_doc.get("axes", {}), axis, stat),
                    "delta_9b_minus_7b": delta_pt,
                    "delta_ci95": _ci(dd),
                    "delta_t11_ci95": [delta_pt - T975_DF11 * sd, delta_pt + T975_DF11 * sd],
                    "delta_loco_jackknife_range": [
                        float(np.nanmin(ld)) if np.isfinite(ld).any() else float("nan"),
                        float(np.nanmax(ld)) if np.isfinite(ld).any() else float("nan"),
                    ],
                    "fire": {
                        k: shared[axis][k]
                        for k in (
                            "symmetric_headline",
                            "n_shared_primary",
                            "n_symmetric_fired",
                            "n_dropped_9b_only",
                            "n_dropped_7b_only",
                        )
                    },
                    "ceiling_cleared": bool(
                        not run9.ceiling_suppressed.get(axis, True)
                        and not run7.ceiling_suppressed.get(axis, True)
                    ),
                }
            )
            axes_used.append(axis)
            d9_all.append(dr9)
            d7_all.append(dr7)
            dd_all.append(dd)
            loco_all.append(ld)
        s9 = np.array([r["s_9b"] for r in rows])
        s7 = np.array([r["s_7b"] for r in rows])
        ax_ix = [parent_axes.index(a) for a in axes_used]
        sp = spearman_block(s9, s7, rng_mc)
        sp_partial = partial_spearman(s9, s7, ct9[ax_ix], ct7[ax_ix])
        cleared = np.array([r["ceiling_cleared"] for r in rows], dtype=bool)
        sp_cleared = (
            spearman_block(s9[cleared], s7[cleared], rng_mc)
            if int(cleared.sum()) >= 3
            else {"rho": float("nan"), "n": int(cleared.sum()), "p": float("nan"), "method": "n<3"}
        )
        stat_tables[stat] = {
            "definition": stats_def[stat],
            "axes": rows,
            "spearman": sp,
            "spearman_partial_changed_tokens": {
                "rho": sp_partial,
                "note": "per-tokenizer covariates: each side's ranks residualized on ITS OWN "
                "tokenizer's mean changed_tokens per axis (convention 16)",
            },
            "spearman_ceiling_cleared": sp_cleared,
        }
        perdraw[stat] = {
            "axes": np.array(axes_used),
            "draws_9b": np.stack(d9_all) if d9_all else np.zeros((0, mult.shape[0])),
            "draws_7b": np.stack(d7_all) if d7_all else np.zeros((0, mult.shape[0])),
            "delta_draws": np.stack(dd_all) if dd_all else np.zeros((0, mult.shape[0])),
            "loco_delta": np.stack(loco_all) if loco_all else np.zeros((0, n_car)),
        }

    # H2 verdict lattice (plan §3) — read (a) obs separation, (b) map-arm direction.
    def band(rho: float) -> str:
        if not np.isfinite(rho):
            return "h2_undetermined"
        if rho >= 0.6:
            return "h2_shared"
        if rho <= 0.2:
            return "h2_falsified"
        return "h2_inconclusive"

    sign_rows = []
    for r in stat_tables["direction_cos"]["axes"]:
        ax = r["axis"]
        if not r["ceiling_cleared"]:
            continue
        k = list(perdraw["direction_cos"]["axes"]).index(ax)
        dr9 = perdraw["direction_cos"]["draws_9b"][k]
        dr7 = perdraw["direction_cos"]["draws_7b"][k]
        ci9v = _ci(dr9)
        ci7v = _ci(dr7)
        stable9 = np.isfinite(ci9v[0]) and (ci9v[0] > 0 or ci9v[1] < 0)
        stable7 = np.isfinite(ci7v[0]) and (ci7v[0] > 0 or ci7v[1] < 0)
        if stable9 and stable7 and np.sign(r["s_9b"]) != np.sign(r["s_7b"]):
            sign_rows.append(ax)
    rho_a = stat_tables["obs_separation_snr"]["spearman"]["rho"]
    rho_b = stat_tables["direction_cos"]["spearman"]["rho"]
    v_a, v_b = band(rho_a), band(rho_b)
    n_signdis = len(sign_rows)
    if v_a == "h2_falsified" or v_b == "h2_falsified" or n_signdis >= 3:
        combined = "h2_falsified"
    elif v_a == "h2_shared" and v_b == "h2_shared":
        combined = "h2_shared"
    else:
        combined = "h2_inconclusive"
    h2 = {
        "primary_h2_7b_arm": resolve_primary_h2_arm([run7.spec.map_arm]),
        "read_a_obs_separation": {"rho": rho_a, "verdict": v_a},
        "read_b_maparm_direction": {"rho": rho_b, "verdict": v_b},
        "sign_disagreement_axes": sign_rows,
        "n_sign_disagreements": n_signdis,
        "sign_rule": "screened to ceiling-cleared axes; a disagreement requires BOTH sides' "
        "bootstrap 95% CIs of the direction cos to exclude 0 (sign stability from the "
        "bootstrap sign distribution, not point signs) with opposite point signs",
        "bands": "rho >= 0.6 shared; rho <= 0.2 falsified; 0.2 < rho < 0.6 inconclusive; "
        ">= 3 screened sign disagreements falsifies regardless",
        "combined_verdict": combined,
        "combined_rule": "falsified if either read falsified or >=3 screened sign "
        "disagreements; shared iff BOTH reads >= 0.6; else inconclusive",
    }

    doc = {
        "layer_pair": {"qwen35_9b": int(lstar), "qwen25_7b": L19},
        "primary_h2_7b_arm": PRIMARY_H2_7B_ARM,
        "map_arms": {"qwen35_9b": run9.spec.map_arm, "qwen25_7b": run7.spec.map_arm},
        "ref_7b_parent": {
            "label": REF_7B_PARENT,
            "role": "SEPARATELY LABELED sensitivity read only — never the primary comparison",
            "commit": ref7b_commit,
            "arm": "arm_779ce (the parent's primary frozen map)",
        },
        "fire_gating": "symmetric: a pair (or vp) non-fired on EITHER model drops from BOTH "
        "sides; axes where either side is compliance-limited fall back to all shared primary "
        "pairs (symmetric_headline=false recorded per axis)",
        "bootstrap": {
            "scheme": "ONE shared 12-carrier resample per draw, BOTH models evaluated on it",
            "B": int(mult.shape[0]),
            "seed": BOOT_SEED,
            "t11_companion": "point +/- t_{0.975,11} * sd(bootstrap delta draws) — convention "
            "15 (G=12 percentile CIs undercover)",
            "loco_jackknife": "12 leave-one-carrier-out point recomputes (weight rows)",
        },
        "stats": stat_tables,
        "h2": h2,
    }
    return doc, perdraw


# ── ref_7b_parent loading ──────────────────────────────────────────────


def load_ref7b_parent(path: Path, commit: str) -> dict:
    """Parent committed minpair_delta.json (schema probed on the issue-2564
    branch at the recorded commit — plan §4.5: freshest commit >= the pin)."""
    doc = json.loads(Path(path).read_text())
    for key in ("axes", "contract", "meta"):
        assert key in doc, (key, "ref7b parent JSON missing a top-level key")
    n_axes = len(doc["axes"])
    assert n_axes == 11, (n_axes, "expected the 11 parent axes")
    logger.info("[an] ref_7b_parent loaded: %d axes at commit %s", n_axes, commit)
    return doc


# ── io + main ──────────────────────────────────────────────────────────


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True))


def _json_sanitize(obj):
    """NaN/inf -> None recursively (JSON round-trip safety)."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        v = obj.item()
        return None if isinstance(v, float) and not math.isfinite(v) else v
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    return obj


def side_meta(run: SideRun) -> dict:
    return {
        "model_tag": run.spec.name,
        "d": run.spec.d,
        "primary_layer": run.spec.primary_layer,
        "twin_layers": list(run.spec.twin_layers),
        "arms": list(run.spec.arms),
        "map_arm": run.spec.map_arm,
        "id_arm": run.spec.id_arm,
        "n_contexts": len(run.st.ctx_ids),
        "n_pairs": run.pa.n,
        "cells": run.st.cells,
        "carriers": run.st.carriers,
        "exclusions": run.st.exclusions,
        "engine_parity": run.engine_parity,
        "identity_cancellation_assert": run.id_check,
        "input_files": run.st.input_files,
    }


def base_contract(cfg: CfgX, run9: SideRun, run7: SideRun) -> dict:
    return {
        "primary_h2_7b_arm": PRIMARY_H2_7B_ARM,
        "null_scheme": {
            "qwen35_9b": {a: v.null_scheme for a, v in run9.views.items()},
            "qwen25_7b": {a: v.null_scheme for a, v in run7.views.items()},
        },
        "null_seed_offsets": NULL_OFFSET,
        "bootstrap": {
            "scheme": "carrier-clustered (resample the 12 carrier clusters with replacement); "
            "ONE shared index matrix serves both sides AND the cross-model battery",
            "query_content": DYADIC_BOOTSTRAP_CONVENTION,
            "B": cfg.b_boot,
            "seed": BOOT_SEED,
            "gsmall_caveat": "G=12 clusters — percentile CIs undercover (convention 8/15; "
            "t11 + LOCO companions ride the cross-model doc)",
        },
        "null": {"B": cfg.b_null, "seed": NULL_SEED},
        "split_half": {"n_splits": cfg.n_splits, "seed": SPLIT_SEED},
        "h1_row_bootstrap_seed": list(H1_BOOT_SEED),
        "orientation_conventions": ORIENTATION_CONVENTIONS,
        "compliance": {
            "headline_threshold_pct": 70,
            "sensitivity_pcts": [50, 90],
            "rule": "headline per-axis reads are FLOOR-GATED (axis_row.floor_met); "
            "compliance-limited axes report NULL headline fields; *_all_values companions "
            "always populated; special axis rows without floor_met are unfiltered",
        },
        "port_source": {
            "script": "scripts/issue2564_analysis.py",
            "pin": PORT_SOURCE_PIN,
        },
    }


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
    t0 = time.time()
    print(
        f"[phase=an_2587] smoke={cfg.smoke} out_dir={cfg.out_dir} b_boot={cfg.b_boot} "
        f"b_null={cfg.b_null} n_splits={cfg.n_splits}",
        flush=True,
    )
    cfg.ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg.perdraw_dir.mkdir(parents=True, exist_ok=True)

    # frozen L* (constraint: READ, never re-argmaxed)
    ls = load_lstar(cfg.sweep_json)
    lstar = ls["lstar"]

    # banks
    bank9 = json.loads(cfg.bank_9b.read_text())
    assert (
        bank9["n_contexts"] == B87.N_CONTEXTS and bank9["n_pairs"] == B87.N_PAIRS
    ) or cfg.smoke, (
        bank9["n_contexts"],
        bank9["n_pairs"],
    )
    for p in bank9["pairs"]:
        assert "changed_tokens" in p, (p["pair_id"], "bank manifest missing changed_tokens")
    bank7_path = (
        cfg.bank_7b
        if cfg.bank_7b is not None
        else resolve_rel(cfg, cfg.in_root_7b, cfg.prefix_2564, "manifests/bank2564_manifest.json")
    )
    bank7 = json.loads(Path(bank7_path).read_text())
    for p in bank7["pairs"]:
        assert "changed_tokens" in p, (p["pair_id"], "parent bank missing changed_tokens")

    # instruction axes = parent bank cells minus query classes (from the banks)
    instr_axes = tuple(
        sorted(
            {
                p["cell"]
                for p in bank7["pairs"]
                if p["cell"] != "query" and p["pair_class"] in ("swap",)
            }
        )
    )
    spec9 = make_spec_9b(lstar, instr_axes)
    spec7 = make_spec_7b(instr_axes)

    assert cfg.manip_9b.exists(), f"9B manipulation check missing: {cfg.manip_9b}"
    assert cfg.manip_7b.exists(), (
        f"7B manipulation check missing: {cfg.manip_7b} — recover it from the parent branch: "
        "git show origin/issue-2564:eval_results/issue_2564/manipulation_check.json"
    )
    fire9 = load_fire(cfg.manip_9b)
    fire7 = load_fire(cfg.manip_7b)

    # shared carrier bootstrap (ONE index matrix; both sides + cross-model)
    rng_boot = np.random.default_rng([BOOT_SEED])
    st9 = load_stores_9b(cfg, bank9, spec9)
    st7 = load_stores_7b(cfg, bank7, spec7)
    assert st9.carriers == st7.carriers, (st9.carriers, st7.carriers)
    n_car = len(st9.carriers)
    if not cfg.smoke:
        assert n_car == 12, st9.carriers
    idx_draws = rng_boot.integers(0, n_car, size=(cfg.b_boot, n_car))
    mult = carrier_multiplicities(idx_draws, n_car)

    # 9B arms: frozen L* ridge payload (apply_map) + iddelta
    ridge_path = (
        cfg.ridge_9b
        if cfg.ridge_9b is not None
        else resolve_rel(cfg, None, cfg.prefix_fits, f"analysis_tensors/ridge_payloads/L{lstar}.pt")
    )
    payload = load_ridge_payload(ridge_path, H_9B, ARM_FRESH9B)
    mapped9 = {
        ARM_FRESH9B: N1M.apply_map(payload, st9.vc[lstar], torch.device("cpu")),
        ARM_IDD9B: st9.vc[lstar],
    }
    ridge_meta = {
        "path": str(ridge_path),
        "bytes": ridge_path.stat().st_size,
        "sha256": _sha256(ridge_path),
        "layer": lstar,
    }

    # 7B arms: unit 4's persisted matched-capacity mapped bank + iddelta
    def _preds7b_file(name: str) -> Path:
        if cfg.preds7b_dir is not None:
            p = cfg.preds7b_dir / name
            assert p.exists(), p
            return p
        from explore_persona_space.orchestrate.hub import stage_hub_file

        target = cfg.stage_dir / cfg.prefix_preds7b / name
        if target.exists():
            return target
        logger.info("[an] staging %s/%s", cfg.prefix_preds7b, name)
        return Path(stage_hub_file(HF_DATA_REPO, f"{cfg.prefix_preds7b}/{name}", target))

    mapped7b_path = _preds7b_file(f"mapped_vc2564_{ARM_7B_MATCHED}_L{L19}.pt")
    m7 = torch.load(mapped7b_path, map_location="cpu", weights_only=False)
    m7_ids = [str(x) for x in m7["context_ids"]]
    m7_of = {cid: i for i, cid in enumerate(m7_ids)}
    missing7 = [cid for cid in st7.ctx_ids if cid not in m7_of]
    assert not missing7, f"matched-7B mapped bank missing contexts: {missing7[:5]}"
    m7_mat = np.asarray(m7["tensor"], dtype=np.float64)[[m7_of[c] for c in st7.ctx_ids]]
    assert m7_mat.shape == (len(st7.ctx_ids), H_7B), m7_mat.shape
    mapped7 = {ARM_7B_MATCHED: m7_mat, ARM_IDD7B: st7.vc[L19]}

    run9 = compute_side(cfg, spec9, bank9, st9, fire9, mapped9, mult, idx_draws)
    _write_json_atomic(
        cfg.ckpt_dir / "battery_qwen35_9b.json",
        _json_sanitize({"meta": side_meta(run9), "axes": run9.axes_out}),
    )
    print("[an] checkpoint battery_qwen35_9b.json written", flush=True)
    run7 = compute_side(cfg, spec7, bank7, st7, fire7, mapped7, mult, idx_draws)
    _write_json_atomic(
        cfg.ckpt_dir / "battery_qwen25_7b.json",
        _json_sanitize({"meta": side_meta(run7), "axes": run7.axes_out}),
    )
    print("[an] checkpoint battery_qwen25_7b.json written", flush=True)

    # H1
    preds9_path = (
        cfg.preds_9b
        if cfg.preds_9b is not None
        else resolve_rel(cfg, None, cfg.prefix_fits, f"analysis_tensors/preds/L{lstar}_preds.pt")
    )
    preds9 = torch.load(preds9_path, map_location="cpu", weights_only=False)
    preds7_path = _preds7b_file(f"test_preds_{ARM_7B_MATCHED}_L{L19}.pt")
    preds7 = torch.load(preds7_path, map_location="cpu", weights_only=False)
    h1_doc, h1_draws = compute_h1(
        preds9, preds7, lstar, cfg.b_boot, np.random.default_rng(list(H1_BOOT_SEED))
    )
    _write_json_atomic(cfg.ckpt_dir / "h1.json", _json_sanitize(h1_doc))
    np.savez(
        cfg.perdraw_dir / "h1_delta_draws.npz",
        delta_draws=h1_draws["delta_draws"],
        r9_draws=h1_draws["r9_draws"],
        r7_draws=h1_draws["r7_draws"],
    )
    print(f"[an] H1 verdict: {h1_doc['verdict']} delta={h1_doc['delta_map']:.4f}", flush=True)

    # ref_7b_parent + cross-model
    ref7b = load_ref7b_parent(cfg.ref7b_parent, cfg.ref7b_parent_commit)
    cm_doc, cm_perdraw = crossmodel_contrasts(
        run9, run7, lstar, mult, ref7b, cfg.ref7b_parent_commit, cfg
    )
    for stat, blk in cm_perdraw.items():
        np.savez(
            cfg.perdraw_dir / f"{stat}.npz",
            axes=np.array([str(a) for a in blk["axes"]]),
            draws_9b=blk["draws_9b"],
            draws_7b=blk["draws_7b"],
            delta_draws=blk["delta_draws"],
            loco_delta=blk["loco_delta"],
        )
    meta_common = {
        "issue": ISSUE,
        "smoke": cfg.smoke,
        "primary_h2_7b_arm": resolve_primary_h2_arm([spec7.map_arm]),
        "lstar": ls,
        "ridge_payload_9b": ridge_meta,
        "ref7b_parent": {"path": str(cfg.ref7b_parent), "commit": cfg.ref7b_parent_commit},
        "engine_parity": {"qwen35_9b": run9.engine_parity, "qwen25_7b": run7.engine_parity},
        "timestamp_utc": datetime.now(UTC).isoformat(),
        **as_metadata_dict(git_provenance(), phase="an-2587"),
    }
    cm_doc["meta"] = {
        **meta_common,
        "perdraw_dir": str(cfg.perdraw_dir),
        "h1_preds": {"qwen35_9b": str(preds9_path), "qwen25_7b": str(preds7_path)},
    }
    _write_json_atomic(cfg.out_dir / "crossmodel_contrasts.json", _json_sanitize(cm_doc))
    print("[an] crossmodel_contrasts.json written", flush=True)

    # merged primary deliverable
    doc = {
        "meta": {
            **meta_common,
            "phase": "an_2587",
            "elapsed_s": round(time.time() - t0, 1),
            "manip_check": {"qwen35_9b": str(cfg.manip_9b), "qwen25_7b": str(cfg.manip_7b)},
        },
        "contract": base_contract(cfg, run9, run7),
        "sides": {
            "qwen35_9b": {
                "meta": side_meta(run9),
                "axes": run9.axes_out,
                "retrieval": run9.retrieval,
                "pilot_placement": pilot_placement_block(run9),
            },
            "qwen25_7b": {
                "meta": side_meta(run7),
                "axes": run7.axes_out,
                "retrieval": run7.retrieval,
            },
        },
        "h1": h1_doc,
        "h2": cm_doc["h2"],
    }
    _write_json_atomic(cfg.out_dir / "minpair_delta_2587.json", _json_sanitize(doc))
    rows = [json.dumps(_json_sanitize(r), sort_keys=True) for r in run9.perpair + run7.perpair]
    with atomic_replace(cfg.out_dir / "perpair_2587.jsonl") as tmp:
        tmp.write_text("\n".join(rows) + "\n")
    print(
        f"[an] wrote {cfg.out_dir / 'minpair_delta_2587.json'} + perpair_2587.jsonl "
        f"({len(rows)} rows) + crossmodel_contrasts.json",
        flush=True,
    )
    print("[phase=done] an_2587 complete", flush=True)
    return 0


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_rc)
