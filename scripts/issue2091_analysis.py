"""#2091 unit C — P4 analysis driver + statistics engine (plan v4 §4.2 P4, §6).

Sequential per-family stage→consume→reap loop (MF-1) on the VM:

  Family A (generic): #1073 ``v_store`` shards + the LMSYS pass-B bundle +
  the wcrung banked capture slice + the new greedy WildChat slice → reduce
  the generic core to compact retained fp32 matrices → run the generic /
  LMSYS legs (R1/R2 generic bars, R4 generic panel, wildchat R3/R5 panels)
  → REAP ``v_store`` + the LMSYS bundle + the wcrung/greedy stores.

  Per-behavior families (×3, sequential): labeling-tar slice via
  ``issue1739_map963k_slice.stream_slice`` (kinds t1/context_end × layers
  14/19/26), greedy-store slices per-file, the parity-probe store → fits
  (unit B's ``issue2091_fits`` wrapper) + batteries + the full-coverage
  capture-parity read → REAP the family's staged inputs.

Staging root: ``/mnt/eps-data/thomasjiralerspong/issue2091_hf_dl/`` (plan
§9 disk row — NEVER ``data/issue_2091/`` (the #681 worktree bind is not
live, so that path lands on ``/``), never ``/tmp``). ``df -P`` is probed at
driver start; free < 30 GB exits with the DESIGNED rc naming the
pre-registered ``cpu-bigmem`` fallback (plan §4.2 MF-1 item 4). Plan §7 G3
(fit pilot): the FIRST fresh ``fit_pool_regime`` call is timed by
``FitTimer`` and a > 2x projection over the §9 P4-fits booking exits
``rc=RC_FIT_WALL`` with ``g3_fit_pilot_gate.json`` persisted — the remedy is
the vectorize check, never a silent descope.

Outputs (§6.5 primary-deliverable globs, all under ``eval_results/issue_2091/``):
``r1_dispersion.json``, ``r2_delta.json``, ``r3_moderators_<behavior>.json``,
``r4_grids.json``, ``r5_polarization.json``, ``capture_parity.json``.

Trigger-dense discipline: this driver handles real-user-corpus text
(LMSYS/WildChat) and behavior-eval completions IN CODE ONLY — no row text is
ever printed/logged; digests (counts, shas, medians) only.

MALLOC_ARENA_MAX cannot be retrofitted in-process (glibc reads it at malloc
init) — the launch prefix carries it (code-style § shared-VM thread caps);
the numeric thread caps below are the in-module backstop.
"""

from __future__ import annotations

import os

# In-module backstop for the shared-VM thread caps (before numpy/torch import).
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "8")

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path

# Canonical dotenv wrapper BEFORE the first heavy import (#847 thread-caps gate);
# also the credential load for the Hub phases (idempotent with _hub()'s call).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402


def _ensure_repo_root_on_syspath() -> Path:
    """Script-mode guard (gotchas.md #823): repo root on sys.path, sentinel-checked."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2091_fits.py").is_file(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

logging.basicConfig(level=logging.INFO, format="[%(levelname)s %(asctime)s] %(message)s")
logger = logging.getLogger("issue2091_analysis")

# ── pins (plan §9/§10/§11) ────────────────────────────────────────────────────
TASK_ID = 2091
SEED = 20910
LAYERS: tuple[int, ...] = (14, 19, 26)
HEADLINE_LAYER = 19
K_ROLLOUTS = 5
LMSYS_K_BANKED = 10  # #1073 stoch10 grain; fixed-seed 5-of-10 subsample (plan §4.2)
B_BOOT_DEFAULT = 2000
B_BOOT_DESCOPED = 500  # pre-registered descope lever (plan §9 battery row)
B_BOOT_PILOT = 50  # timed sub-battery gate (plan §9: extrapolated wall <= 2x 0.5h)
BATTERY_WALL_BUDGET_S = 3600.0  # 2x the 0.5 h booking
MID_LO, MID_HI = 25.0, 75.0  # middling band (body Result 5: mu in [25, 75])
SEVERE_TAIL = -0.02  # R2-b severe-tail threshold (plan §6)
BEHAVIORS: tuple[str, ...] = ("sycophancy", "hallucination", "evil")
FAMILIES: tuple[str, ...] = ("generic",) + BEHAVIORS

HF_PREFIX = "issue2091_decode"
DATA_REPO = "superkaiba1/explore-persona-space-data"
CTXMAP_PREFIX = "issue1739_ctxmap"
V_STORE_PREFIX = "issue1073_decode_regime/analysis_tensors/v_store"
V_STORE_REVISION = "fb4fe90fdd836ba2efd896b90c17e6b42f143d21"  # #1073 pin (greedy_cloud STORE_REV)
LMSYS_BUNDLE_PATH = "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
WCRUNG_STORE_PREFIX = f"{CTXMAP_PREFIX}/wildchat_rung/capture_store"
MAP963K_NPZ_PATH = f"{CTXMAP_PREFIX}/analysis_tensors/maps/context_end__ufull.npz"
PER_DRAW_PREFIX = f"{CTXMAP_PREFIX}/judge_reliability"

DEFAULT_STAGING_ROOT = Path("/mnt/eps-data/thomasjiralerspong/issue2091_hf_dl")
DEFAULT_OUT_ROOT = REPO_ROOT / "eval_results" / "issue_2091"
BANKED_DV_DIR = REPO_ROOT / "eval_results" / "issue_1739" / "dv_dataset"

# Designed exit codes (never anonymous rc=1; pilot-gate routing family, #1415).
RC_DISK_FALLBACK = 8  # staging headroom below floor -> cpu-bigmem fallback
RC_FIT_WALL = 9  # G3 fit pilot: projected fits wall > 2x the §9 P4-fits booking
CPU_BIGMEM_FALLBACK_CMD = (
    "uv run python scripts/dispatch_issue.py --issue 2091 --intent cpu-bigmem --boot-disk-gb 120"
)
START_FLOOR_GB = 30.0  # 1.5x the declared 20 GB co-resident peak (plan §9)
FAMILY_FLOOR_GB = {"generic": 22.0, "sycophancy": 12.0, "hallucination": 12.0, "evil": 12.0}
FIT_WALL_BUDGET_S = 3600.0  # plan §9 P4-fits row booking (1.0 h)
FIT_TOTAL_CALLS = 27  # 9 pools x 3 regimes fit_pool_regime wrapper calls (plan §9 P4-fits)

# Rung-job -> family map (behavior families own their rung-jobs; wildchat = generic).
FAMILY_JOBS = {
    "generic": ("wildchat",),
    "sycophancy": ("syc_train", "syc_aita"),
    "hallucination": ("hal_train", "hal_nqopen", "hal_simpleqa"),
    "evil": ("evil_train", "evil_hhrt", "evil_toxicchat"),
}
REGIMES: tuple[str, ...] = ("greedy", "avg_k5", "single")
# R3 commonality reads ONE pinned regime per map family (never dict-order-dependent):
# avg_k5 matches the commonality DV reference (dv_avg) + the oracle's eval_v5_mean.
COMMONALITY_MAP_REGIME = "avg_k5"


# ── deferred imports (heavy / sibling-script; import-check executes all) ────
def _fits2091():
    from scripts import issue2091_fits as f

    return f


def _stage2091():
    from scripts import issue2091_stage_contexts as s

    return s


def _store_io():
    from explore_persona_space.experiments.issue_1739 import store_io

    return store_io


def _judging():
    from explore_persona_space.experiments.issue_1739 import judging

    return judging


def _hub():
    from explore_persona_space.orchestrate import env as _env

    _env.load_dotenv()
    from explore_persona_space.orchestrate import hub

    return hub


def _slice_mod():
    from scripts import issue1739_map963k_slice as m

    return m


def _cap1073():
    from scripts import issue1073_capture as c

    return c


def _common1073():
    from scripts import issue1073_common as c

    return c


def _provenance_meta() -> dict:
    from explore_persona_space.orchestrate import provenance

    return provenance.as_metadata_dict(provenance.git_provenance())


# ── small utils ───────────────────────────────────────────────────────────────
def write_json_atomic(path: Path, obj) -> None:
    """Atomic JSON write (tmp + os.replace), parents created."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return (
            np.where(np.isfinite(o.astype(np.float64)), o, None).tolist()
            if o.dtype.kind == "f"
            else o.tolist()
        )
    raise TypeError(f"not JSON-serializable: {type(o)}")


def stable_hash64(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "big")


def rng_for(name: str) -> np.random.Generator:
    return np.random.default_rng([SEED, stable_hash64(name)])


def rankdata_avg(v: np.ndarray) -> np.ndarray:
    """Average ranks with ties (scipy-free; judged DVs are tie-heavy)."""
    v = np.asarray(v, dtype=np.float64)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(v.size, dtype=np.float64)
    sv = v[order]
    i = 0
    while i < v.size:
        j = i
        while j + 1 < v.size and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho (average-rank ties); NaN pairs dropped; NaN when n<3."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3:
        return float("nan")
    rx, ry = rankdata_avg(x[keep]), rankdata_avg(y[keep])
    rx -= rx.mean()
    ry -= ry.mean()
    den = float(np.sqrt((rx**2).sum() * (ry**2).sum()))
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def summarize(vals: np.ndarray) -> dict:
    """Distribution digest: n / median / p5 / p95 / min / mean."""
    v = np.asarray(vals, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n": 0, "median": None, "p5": None, "p95": None, "min": None, "mean": None}
    return {
        "n": int(v.size),
        "median": float(np.median(v)),
        "p5": float(np.percentile(v, 5)),
        "p95": float(np.percentile(v, 95)),
        "min": float(v.min()),
        "mean": float(v.mean()),
    }


def iter_jsonl(path: Path):
    """Text-mode JSONL iteration (never .splitlines() — gotchas.md U+2028)."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


# ── group bootstrap engine (vectorized; plan §6 cluster bootstrap) ────────────
class GroupBootstrap:
    """Vectorized cluster bootstrap over group labels (+ optional 2nd axis).

    Groups are resampled with replacement (multinomial counts); per-context
    weights = group multiplicity (PRODUCT across axes for the evil two-way
    variant — reduces exactly to one-way when the second axis is constant:
    a single question cluster's multinomial count is identically its total,
    a constant factor that cancels in every weighted statistic).
    Every statistic is a weighted reduction batched over the B draws
    (vectorize-many-cell-fits.md: one GEMM / one sorted-cumsum per stat).
    Spearman-type draws use GLOBAL average ranks as scores (standard
    Pearson-on-ranks bootstrap approximation; ranks fixed at full sample).
    """

    def __init__(
        self,
        groups: Sequence[str],
        b: int,
        seed_key: str,
        groups2: Sequence[str] | None = None,
    ) -> None:
        self.n = len(groups)
        _, gidx = np.unique(np.asarray(groups, dtype=object), return_inverse=True)
        g = int(gidx.max()) + 1 if self.n else 1
        rng = rng_for(f"boot::{seed_key}")
        counts = rng.multinomial(g, np.full(g, 1.0 / g), size=b)  # (B, G)
        w = counts[:, gidx].astype(np.float64) if self.n else np.zeros((b, 0))
        self.axes = {"groups": g}
        if groups2 is not None:
            assert len(groups2) == self.n, (len(groups2), self.n)
            _, qidx = np.unique(np.asarray(groups2, dtype=object), return_inverse=True)
            q = int(qidx.max()) + 1
            counts2 = rng.multinomial(q, np.full(q, 1.0 / q), size=b)
            w = w * counts2[:, qidx].astype(np.float64)
            self.axes["groups2"] = q
        self.w = w  # (B, n)
        self.b = b

    def _mask_weights(self, mask: np.ndarray | None) -> np.ndarray:
        if mask is None:
            return self.w
        return self.w * np.asarray(mask, dtype=np.float64)[None, :]

    def mean(self, vals: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        raw = np.asarray(vals, dtype=np.float64)
        v = np.nan_to_num(raw, nan=0.0)
        finite = np.isfinite(raw).astype(np.float64)
        w = self._mask_weights(mask) * finite[None, :]
        tot = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return (w @ v) / tot

    def median(self, vals: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        v = np.asarray(vals, dtype=np.float64)
        keep = np.isfinite(v)
        if mask is not None:
            keep &= np.asarray(mask, dtype=bool)
        v = v[keep]
        if v.size == 0:
            return np.full(self.b, np.nan)
        w = self.w[:, keep]
        order = np.argsort(v, kind="stable")
        sv = v[order]
        sw = w[:, order]
        cum = np.cumsum(sw, axis=1)
        tot = cum[:, -1:]
        with np.errstate(invalid="ignore"):
            idx = np.argmax(cum >= 0.5 * tot, axis=1)
        out = sv[idx]
        out[tot[:, 0] <= 0] = np.nan
        return out

    def corr(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Weighted Pearson per draw (rank-transform inputs upstream for Spearman)."""
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        keep = np.isfinite(x) & np.isfinite(y)
        if mask is not None:
            keep &= np.asarray(mask, dtype=bool)
        if keep.sum() < 3:
            return np.full(self.b, np.nan)
        x, y = x[keep], y[keep]
        w = self.w[:, keep]
        sw = w.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx = (w @ x) / sw
            my = (w @ y) / sw
            exx = (w @ (x * x)) / sw - mx**2
            eyy = (w @ (y * y)) / sw - my**2
            exy = (w @ (x * y)) / sw - mx * my
            return exy / np.sqrt(exx * eyy)

    @staticmethod
    def ci(draws: np.ndarray, alpha: float = 0.05) -> list[float] | None:
        d = np.asarray(draws, dtype=np.float64)
        d = d[np.isfinite(d)]
        if d.size == 0:
            return None
        return [
            float(np.percentile(d, 100 * alpha / 2)),
            float(np.percentile(d, 100 * (1 - alpha / 2))),
        ]


def holm_adjusted_cis(diff_draws: dict[str, np.ndarray], alpha: float = 0.05) -> dict:
    """Holm-adjusted bootstrap CIs + p-values for a family of paired contrasts.

    p per contrast = two-sided bootstrap sign probability (with the +2/B
    resolution floor); Holm step-down over the family; each contrast's CI is
    ALSO reported at its Holm-adjusted level so the CI read matches the test
    (plan §6: generic-vs-trait difference tested directly, Holm-adjusted CIs).
    """
    names = list(diff_draws)
    pvals: dict[str, float] = {}
    for name in names:
        d = np.asarray(diff_draws[name], dtype=np.float64)
        d = d[np.isfinite(d)]
        if d.size == 0:
            pvals[name] = float("nan")
            continue
        frac = min((d <= 0).mean(), (d >= 0).mean())
        pvals[name] = float(min(1.0, 2 * frac + 2.0 / d.size))
    m = len(names)
    order = sorted(names, key=lambda k: (np.isnan(pvals[k]), pvals[k]))
    out = {}
    running_max = 0.0
    for rank, name in enumerate(order):
        adj_alpha = alpha / (m - rank)
        p_adj = None
        if np.isfinite(pvals[name]):
            running_max = max(running_max, min(1.0, pvals[name] * (m - rank)))
            p_adj = running_max
        out[name] = {
            "p_holm": p_adj,
            "ci_holm": GroupBootstrap.ci(diff_draws[name], alpha=adj_alpha),
            "ci95": GroupBootstrap.ci(diff_draws[name]),
            "median": float(np.nanmedian(diff_draws[name])),
        }
    return out


# ── pure statistics (plan §6) ─────────────────────────────────────────────────
def _unit_rows(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    n = np.where(n > 0, n, 1.0)
    return v / n


def pairwise_cos_dispersion(v: np.ndarray) -> np.ndarray:
    """R1: per-context mean pairwise cosine DISTANCE among K rows. v: (n, K, d)."""
    u = _unit_rows(np.asarray(v, dtype=np.float64))
    g = np.einsum("nkd,njd->nkj", u, u)
    k = u.shape[1]
    off = (g.sum(axis=(1, 2)) - np.trace(g, axis1=1, axis2=2)) / (k * (k - 1))
    return 1.0 - off


def delta_ctx(g: np.ndarray, v: np.ndarray) -> dict[str, np.ndarray]:
    """R2 matched-reference Delta (LOO-symmetric, plan §6).

    Delta_ctx = mean_j cos(g, LOO_j) - mean_j cos(v_j, LOO_j); both legs share
    the SAME LOO reference sets by design, so reference noise cancels in the
    DIFFERENCE (the plan §6 noise-structure declaration — the paired
    construction is the registered fix, not disjoint halves).
    """
    g = np.asarray(g, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    _n, k, _d = v.shape
    tot = v.sum(axis=1, keepdims=True)
    loo = (tot - v) / (k - 1)  # (n, K, d): LOO_j
    lu = _unit_rows(loo)
    gu = _unit_rows(g)
    vu = _unit_rows(v)
    cos_g = np.einsum("nd,nkd->nk", gu, lu)
    cos_v = np.einsum("nkd,nkd->nk", vu, lu)
    kmean_u = _unit_rows(v.mean(axis=1))
    return {
        "delta": cos_g.mean(axis=1) - cos_v.mean(axis=1),
        "cos_g_loo_mean": cos_g.mean(axis=1),
        "cos_v_loo_mean": cos_v.mean(axis=1),
        "cos_g_kmean": np.einsum("nd,nd->n", gu, kmean_u),
    }


def exchangeability_ranks(g: np.ndarray, v: np.ndarray) -> np.ndarray:
    """R2 structural null: rank of greedy among the K+1 pooled rollouts.

    Statistic per member i: mean cosine to the OTHER K members — symmetric in
    the pooled set, so under exchangeability the greedy member's rank is
    uniform on 1..K+1. Exact ties get average ranks. Returns per-context ranks.
    """
    pool = np.concatenate([np.asarray(g, dtype=np.float64)[:, None, :], np.asarray(v)], axis=1)
    u = _unit_rows(pool)
    gr = np.einsum("nkd,njd->nkj", u, u)
    k1 = u.shape[1]
    s = (gr.sum(axis=2) - 1.0) / (k1 - 1)  # mean cos to others, per member
    greedy = s[:, :1]
    below = (s < greedy).sum(axis=1)
    ties = (s == greedy).sum(axis=1)  # includes self
    return below + (ties + 1) / 2.0


def jackknife_delta_band(g: np.ndarray, v: np.ndarray) -> dict:
    """R2 draw-jackknife: median Delta recomputed dropping each draw (K reps)."""
    k = v.shape[1]
    medians = []
    for drop in range(k):
        keep = [j for j in range(k) if j != drop]
        medians.append(float(np.median(delta_ctx(g, v[:, keep, :])["delta"])))
    return {"drop_one_medians": medians, "band": [min(medians), max(medians)]}


def disjoint_half_agreement(v: np.ndarray, seed_key: str) -> np.ndarray:
    """R2-c/R3 noise reference: cos(mean of 2 draws, mean of other 3) at K=5.

    Approximately (NOT exactly) noise-matched at K=5 — the 2-vs-3 split is the
    plan A2 caveat; gaps are weighed against the jackknife band downstream.
    """
    v = np.asarray(v, dtype=np.float64)
    n, k, _ = v.shape
    rng = rng_for(f"disjoint::{seed_key}")
    ha = np.zeros(n)
    for i in range(n):
        perm = rng.permutation(k)
        a, b = perm[: k // 2], perm[k // 2 :]
        ha[i] = float(np.dot(_unit_rows(v[i, a].mean(axis=0)), _unit_rows(v[i, b].mean(axis=0))))
    return ha


def per_rollout_score_matrix(rows: list[dict], k: int = K_ROLLOUTS) -> tuple[np.ndarray, list[str]]:
    """(n, K) score matrix (NaN where dropped) + context ids, from DV rows."""
    fits = _fits2091()
    mat = np.full((len(rows), k), np.nan)
    ids = []
    for i, row in enumerate(rows):
        ids.append(str(row["context_id"]))
        prs = row.get("per_rollout_scores") or {}
        for kk, s in fits.parse_per_rollout_scores(prs).items():
            if kk < k and s is not None:
                mat[i, kk] = float(s)
    return mat, ids


def variance_components(scores: np.ndarray) -> dict:
    """R3: within/between SD with the Var_within/K correction (plan §6)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(scores, axis=1)
        kept = np.isfinite(scores).sum(axis=1)
        var_within = np.nanvar(scores, axis=1, ddof=0)
    ok = kept >= 2
    var_between_raw = float(np.nanvar(mu[ok], ddof=1)) if ok.sum() >= 3 else float("nan")
    mean_within = float(np.nanmean(var_within[ok])) if ok.any() else float("nan")
    mean_k = float(np.mean(kept[ok])) if ok.any() else float("nan")
    var_between_corr = var_between_raw - (mean_within / mean_k if mean_k > 0 else np.nan)
    rel_kmean = var_between_corr / var_between_raw if var_between_raw > 0 else float("nan")
    return {
        "n_contexts": int(ok.sum()),
        "between_sd_raw": float(np.sqrt(var_between_raw)) if var_between_raw >= 0 else None,
        "within_sd_mean": float(np.sqrt(mean_within)) if mean_within >= 0 else None,
        "between_sd_corrected": (
            float(np.sqrt(var_between_corr))
            if np.isfinite(var_between_corr) and var_between_corr > 0
            else 0.0
        ),
        "var_between_raw": var_between_raw if np.isfinite(var_between_raw) else None,
        "var_within_mean": mean_within if np.isfinite(mean_within) else None,
        "mean_k_kept": mean_k if np.isfinite(mean_k) else None,
        "reliability_kmean": (
            float(np.clip(rel_kmean, 0.0, 1.0)) if np.isfinite(rel_kmean) else None
        ),
    }


def column_ceilings(
    scores: np.ndarray, judge_draw_var: float | None, n_judge_draws: int = 3
) -> dict:
    """Per-regime-column reliability ceilings (plan A3: each column vs its OWN).

    Variance decomposition: observed per-rollout score = true context effect
    + rollout effect + judge noise/n_draws. ``judge_draw_var`` (per-draw
    judge variance from the REALIZED banked draw matrices) splits
    rollout-vs-judge inside the measured within-context variance; None
    degrades to judge_var=0 with the ceilings then slightly optimistic for
    the 1-completion columns (noted by the caller).
    """
    vc = variance_components(scores)
    if vc["var_between_raw"] is None or vc["var_within_mean"] is None:
        return {"components": vc, "ceil_greedy": None, "ceil_avg_k5": None, "ceil_single": None}
    var_true = max(
        vc["var_between_raw"] - vc["var_within_mean"] / max(vc["mean_k_kept"] or 1, 1e-9), 0.0
    )
    jv = float(judge_draw_var or 0.0)
    var_rollout = max(vc["var_within_mean"] - jv / n_judge_draws, 0.0)

    def ceiling(m_rollouts: float) -> float | None:
        var_obs = var_true + var_rollout / m_rollouts + jv / (m_rollouts * n_judge_draws)
        if not np.isfinite(var_obs) or var_obs <= 0:
            return None
        val = np.sqrt(np.clip(var_true / var_obs, 0.0, 1.0))
        return float(val) if np.isfinite(val) else None

    return {
        "var_true_between": var_true,
        "var_rollout": var_rollout,
        "judge_draw_var": jv if judge_draw_var is not None else None,
        "ceil_greedy": ceiling(1.0),
        "ceil_avg_k5": ceiling(float(vc["mean_k_kept"] or K_ROLLOUTS)),
        "ceil_single": ceiling(1.0),
        "components": vc,
    }


def polarization_stats(scores: np.ndarray, ids: list[str]) -> dict:
    """R5: per-context SD-vs-mean vs the sqrt(mu(100-mu)) ceiling + f_mid/q/g_pol.

    f_mid = fraction of a MIDDLING context's draws that are themselves
    middling (mu and draws in [25, 75]); q_pol = mean f_mid - 0.4;
    g_pol = mean f_mid - 0.5 (plan §3 verdict quantities).
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(scores, axis=1)
        sd = np.sqrt(np.nanvar(scores, axis=1, ddof=0))
    kept = np.isfinite(scores).sum(axis=1)
    ok = kept >= 2
    ceil = np.sqrt(np.clip(mu * (100.0 - mu), 0.0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        p = np.where(ceil > 0, sd / ceil, np.nan)
    middling_ctx = ok & (mu >= MID_LO) & (mu <= MID_HI)
    in_band = (scores >= MID_LO) & (scores <= MID_HI)
    with np.errstate(invalid="ignore", divide="ignore"):
        f_mid = np.where(kept > 0, in_band.sum(axis=1) / np.maximum(kept, 1), np.nan)
    mean_f_mid = float(np.nanmean(f_mid[middling_ctx])) if middling_ctx.any() else float("nan")
    return {
        "context_ids": ids,
        "mu": mu,
        "sd": sd,
        "p": p,
        "kept": kept,
        "middling_mask": middling_ctx,
        "f_mid": f_mid,
        "n_middling": int(middling_ctx.sum()),
        "mean_f_mid": mean_f_mid if np.isfinite(mean_f_mid) else None,
        "q_pol": (mean_f_mid - 0.4) if np.isfinite(mean_f_mid) else None,
        "g_pol": (mean_f_mid - 0.5) if np.isfinite(mean_f_mid) else None,
    }


def commonality(y: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> dict:
    """Rank-space commonality decomposition over two predictors (plan R3).

    All three variables average-rank-transformed (Spearman-as-Pearson on
    ranks); 2-predictor commonality algebra: unique_i = R2_full - r_other^2,
    shared = R2_full - unique1 - unique2 (may be NEGATIVE => suppression —
    reported with all raw correlation signs, per the plan's guardrails).
    """
    y = np.asarray(y, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    x2 = np.asarray(x2, dtype=np.float64)
    keep = np.isfinite(y) & np.isfinite(x1) & np.isfinite(x2)
    if keep.sum() < 5:
        return {"n": int(keep.sum()), "r2_full": None}
    ry = rankdata_avg(y[keep])
    r1v = rankdata_avg(x1[keep])
    r2v = rankdata_avg(x2[keep])

    def _corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        den = np.sqrt((a**2).sum() * (b**2).sum())
        return float((a * b).sum() / den) if den > 0 else float("nan")

    r_y1, r_y2, r_12 = _corr(ry, r1v), _corr(ry, r2v), _corr(r1v, r2v)
    denom = 1.0 - r_12**2
    if denom <= 1e-12:
        r2_full = max(r_y1**2, r_y2**2)
    else:
        r2_full = (r_y1**2 + r_y2**2 - 2 * r_y1 * r_y2 * r_12) / denom
    unique1 = r2_full - r_y2**2
    unique2 = r2_full - r_y1**2
    shared = r2_full - unique1 - unique2
    return {
        "n": int(keep.sum()),
        "r2_full": float(r2_full),
        "unique_x1": float(unique1),
        "unique_x2": float(unique2),
        "shared": float(shared),
        "suppression": bool(shared < 0),
        "signs": {"r_y_x1": r_y1, "r_y_x2": r_y2, "r_x1_x2": r_12},
    }


def split_half_reliability(per_half_a: np.ndarray, per_half_b: np.ndarray) -> float | None:
    """Aligned split-half + Spearman-Brown (llm-judging rule 21).

    The halves are the SAME rollout-index partition for every context
    (rollouts {0,2,4} vs {1,3} — item-ALIGNED across contexts by construction).
    """
    r = spearman(per_half_a, per_half_b)
    if not np.isfinite(r):
        return None
    r = max(min(r, 0.999999), -0.999999)
    return float(2 * r / (1 + r))


def r2_score_rows(pred: np.ndarray, y: np.ndarray) -> float:
    """Held-out R^2 (variance-weighted over dims), matching #1739 conventions."""
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0, keepdims=True)) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


# ── staging helpers (MF-1) ────────────────────────────────────────────────────
def df_report(paths: Iterable[str]) -> str:
    try:
        return subprocess.run(
            ["df", "-P", *paths], capture_output=True, text=True, check=False
        ).stdout.strip()
    except OSError as exc:  # pragma: no cover
        return f"df failed: {exc}"


def assert_family_headroom(staging_root: Path, family: str) -> float:
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    return assert_out_root_headroom(
        staging_root, FAMILY_FLOOR_GB[family], phase=f"issue2091-p4-{family}"
    )


def probe_disk(staging_root: Path, *, floor_gb: float = START_FLOOR_GB) -> None:
    """Plan §4.2 MF-1 item 4: designed halt (rc=8) below the staging floor."""
    logger.info("[disk] df -P report:\n%s", df_report(["/", str(staging_root.parent)]))
    bind = subprocess.run(
        ["findmnt", "--mountpoint", str(REPO_ROOT / ".claude" / "worktrees")],
        capture_output=True,
        text=True,
        check=False,
    )
    logger.info("[disk] worktree bind live: %s", "yes" if bind.stdout.strip() else "no")
    staging_root.mkdir(parents=True, exist_ok=True)
    st = os.statvfs(staging_root)
    free_gb = st.f_bavail * st.f_frsize / 1e9
    if free_gb < floor_gb:
        print(
            f"[issue2091-p4] DESIGNED HALT rc={RC_DISK_FALLBACK}: staging free "
            f"{free_gb:.1f} GB < floor {floor_gb:.1f} GB at {staging_root}. "
            f"Pre-registered fallback (plan §9): route P4 to cpu-bigmem via:\n"
            f"  {CPU_BIGMEM_FALLBACK_CMD}",
            flush=True,
        )
        sys.exit(RC_DISK_FALLBACK)
    logger.info("[disk] staging %s free %.1f GB (floor %.1f)", staging_root, free_gb, floor_gb)


def stage_prefix_files(prefix: str, dest: Path, *, revision: str, keep) -> int:
    """Scoped listing + per-file staged download of basenames passing ``keep``.

    The #833 recipe: server-side ``list_repo_tree(path_in_repo=prefix)`` via
    ``hub.list_hf_files_under_path`` + per-file ``hub.stage_hub_file`` (atomic,
    retried). Files land FLAT under ``dest`` (store shards are basename-keyed).
    """
    from huggingface_hub import HfApi

    hub = _hub()
    api = HfApi()
    rels = hub.list_hf_files_under_path(
        api, DATA_REPO, prefix, repo_type="dataset", revision=revision
    )
    picked = [r for r in rels if keep(r.rsplit("/", 1)[-1])]
    if not picked:
        raise FileNotFoundError(f"no files matching filter under {DATA_REPO}@{revision}:{prefix}")
    n_new = 0
    for rel in picked:
        target = dest / rel.rsplit("/", 1)[-1]
        if not target.is_file():
            hub.stage_hub_file(DATA_REPO, rel, target, repo_type="dataset", revision=revision)
            n_new += 1
    logger.info("[stage] %s: %d files (%d new) -> %s", prefix, len(picked), n_new, dest)
    return len(picked)


def store_basename_keep(kinds: tuple[str, ...], layers: tuple[int, ...]):
    """Basename filter for store_io-layout shard dirs (kinds x layers + indexes)."""
    store_io = _store_io()

    def _keep(name: str) -> bool:
        if name.startswith(("row_index", "manifest", "_manifest", "_capture_manifest")):
            return True
        return store_io._wanted_basename(name, kinds, layers)

    return _keep


def reap(path: Path) -> None:
    """Fail-loud between-family reap (MF-1); one log line on every branch."""
    if not path.exists():
        logger.info("[reap] absent (nothing to reap): %s", path)
        return
    if path.is_dir():
        shutil.rmtree(path)  # fail-loud: no ignore_errors
    else:
        path.unlink()
    logger.info("[reap] removed %s", path)


def stage_contexts_tree(staging_root: Path, revision: str) -> Path:
    """Stage issue2091_decode/contexts (manifest + per-job shards + probes)."""
    dest = staging_root / "contexts"
    done = dest / "_staged.json"
    if done.is_file():
        return dest
    from huggingface_hub import HfApi

    hub = _hub()
    api = HfApi()
    prefix = f"{HF_PREFIX}/contexts"
    rels = hub.list_hf_files_under_path(
        api, DATA_REPO, prefix, repo_type="dataset", revision=revision
    )
    if not rels:
        raise FileNotFoundError(f"contexts tree absent under {DATA_REPO}:{prefix}")
    for rel in rels:
        target = dest / rel[len(prefix) + 1 :]
        if not target.is_file():
            hub.stage_hub_file(DATA_REPO, rel, target, repo_type="dataset", revision=revision)
    write_json_atomic(done, {"n_files": len(rels), "revision": revision})
    logger.info("[stage] contexts tree: %d files -> %s", len(rels), dest)
    return dest


def load_job_contexts(contexts_dir: Path, job: str) -> list[dict]:
    stage = _stage2091()
    return stage.load_shard_rows(contexts_dir / job, "ctx")


# ── store access ──────────────────────────────────────────────────────────────
class StoreView:
    """Per-context view over a store_io-layout store (t1/context_end x layers)."""

    def __init__(self, store_dir: Path, kinds=("t1", "context_end"), layers=LAYERS) -> None:
        store_io = _store_io()
        self.arrays, self.meta = store_io.load_summaries(store_dir, tuple(kinds), tuple(layers))
        self.layers = tuple(layers)
        self.rows_by_ctx: dict[str, list[int]] = {}
        for i, m in enumerate(self.meta):
            self.rows_by_ctx.setdefault(str(m.get("context_id")), []).append(i)
        for idxs in self.rows_by_ctx.values():
            idxs.sort(key=lambda i: int(self.meta[i].get("rollout_k") or 0))

    def context_ids(self) -> list[str]:
        return list(self.rows_by_ctx)

    def vc(self, cids: Sequence[str], layer: int) -> np.ndarray:
        """Context vectors (context_end; first row per context) -> (n, d) f32."""
        arr = self.arrays[("context_end", layer)]
        idx = np.array([self.rows_by_ctx[c][0] for c in cids], dtype=int)
        return arr[idx].astype(np.float32)

    def t1_first(self, cids: Sequence[str], layer: int) -> np.ndarray:
        arr = self.arrays[("t1", layer)]
        idx = np.array([self.rows_by_ctx[c][0] for c in cids], dtype=int)
        return arr[idx].astype(np.float32)

    def t1_k(self, cids: Sequence[str], layer: int, k_per_ctx: dict[str, int]) -> np.ndarray:
        """One t1 row per context at that context's picked rollout index."""
        arr = self.arrays[("t1", layer)]
        idx = []
        for c in cids:
            rows = self.rows_by_ctx[c]
            ks = {int(self.meta[i].get("rollout_k") or 0): i for i in rows}
            k = int(k_per_ctx[c])
            if k not in ks:
                raise KeyError(f"context {c}: rollout_k={k} absent from store rows {sorted(ks)}")
            idx.append(ks[k])
        return arr[np.array(idx, dtype=int)].astype(np.float32)

    def t1_stack(self, cids: Sequence[str], layer: int, k: int = K_ROLLOUTS) -> np.ndarray:
        """(n, K, d) f32 per-rollout t1 stack; fails loud on missing rollouts."""
        arr = self.arrays[("t1", layer)]
        out = np.empty((len(cids), k, arr.shape[1]), dtype=np.float32)
        for j, c in enumerate(cids):
            rows = self.rows_by_ctx[c]
            if len(rows) < k:
                raise ValueError(f"context {c}: {len(rows)} rollout rows < K={k}")
            out[j] = arr[np.array(rows[:k], dtype=int)]
        return out


# ── banked / greedy DV access ────────────────────────────────────────────────
def load_banked_dv(behavior: str) -> list[dict]:
    p = BANKED_DV_DIR / behavior / "labeling.json"
    if not p.is_file():
        raise FileNotFoundError(f"banked DV missing: {p}")
    return json.loads(p.read_text())["rows"]


def load_wcrung_dv(behavior: str, staging_root: Path, revision: str) -> list[dict]:
    stage = _stage2091()
    rows, _check = stage.load_wcrung_labeling(
        behavior, revision=revision, stage_dir=staging_root / "wcrung_dv"
    )
    return rows


def load_greedy_dv(out_root: Path, behavior: str) -> dict:
    p = out_root / "greedy_dv" / f"{behavior}.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"greedy DV missing: {p} — P3 (issue2091_judge.py) must complete before this leg"
        )
    return json.loads(p.read_text())


def greedy_dv_by_ctx(payload: dict, *, wildchat_graded: bool = False) -> dict[str, float]:
    rows = payload.get("wildchat_graded_rows") if wildchat_graded else payload["rows"]
    return {str(r["context_id"]): r["dv"] for r in (rows or []) if r.get("dv") is not None}


def judge_draw_variance(behavior: str) -> float | None:
    """Per-draw judge variance from the banked draw matrix (plan A4 footnote).

    REALIZED-matrix rows only; per-behavior coverage (evil 70%) is stated by
    the caller wherever the footnote is used. The ``.npy`` matrices are
    NOT in git (repo-wide ignore) — a sparse worktree lacks them, so fall
    back to the main checkout (git-common-dir parent); absent everywhere,
    degrade to None (ceilings then omit the judge split, noted downstream).
    """
    rel = Path("eval_results") / "issue_1739" / "judge_reliability" / f"draw_matrix_{behavior}.npy"
    p = REPO_ROOT / rel
    if not p.is_file():
        common = subprocess.run(
            ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=False,
            cwd=REPO_ROOT,
        ).stdout.strip()
        if common:
            p = Path(common).parent / rel
    if not p.is_file():
        return None
    m = np.asarray(np.load(p, mmap_mode="r"), dtype=np.float64)
    ok = np.isfinite(m).sum(axis=1) >= 2
    if ok.sum() < 10:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return float(np.nanmean(np.nanvar(m[ok], axis=1, ddof=1)))


# ── S1 / S4 single-draw dispositions ─────────────────────────────────────────
def s1_picks(rows: list[dict]) -> dict[str, dict]:
    """Fixed-seed single-draw picks per context (unit B's S1; plan §4.2)."""
    fits = _fits2091()
    out = {}
    for r in rows:
        prs = r.get("per_rollout_scores")
        if prs is None:
            continue
        pick = fits.s1_single_draw_pick(str(r["context_id"]), prs)
        out[pick.context_id] = {"k": pick.k, "dv": pick.score, "dv_included": pick.dv_included}
    return out


def stage_per_draw_tables(staging_root: Path, revision: str) -> list[Path]:
    """Stage the banked hallucination per-draw abstain tables (S4 inputs)."""
    manifest = json.loads(
        (
            REPO_ROOT
            / "eval_results"
            / "issue_1739"
            / "judge_reliability"
            / "per_draw_manifest.json"
        ).read_text()
    )
    shards = manifest["hallucination"]["shards"]
    dest = staging_root / "per_draw_hallucination"
    hub = _hub()
    paths = []
    for rel in shards:
        base = str(rel).rsplit("/", 1)[-1]
        target = dest / base
        if not target.is_file():
            hub.stage_hub_file(
                DATA_REPO,
                f"{PER_DRAW_PREFIX}/{base}",
                target,
                repo_type="dataset",
                revision=revision,
            )
        paths.append(target)
    return paths


def stage_packed_lookup(
    behavior: str, staging_root: Path, revision: str, wanted_cids: set[str]
) -> dict[tuple[str, int], dict]:
    """(context_id, rollout_k) -> {completion, answer_aliases} for S4 (text in code only)."""
    stage = _stage2091()
    hub = _hub()
    dest = staging_root / f"packed_{behavior}"
    lookup: dict[tuple[str, int], dict] = {}
    for rel in stage.packed_shard_paths(behavior, revision=revision):
        target = dest / rel.rsplit("/", 1)[-1]
        if not target.is_file():
            hub.stage_hub_file(DATA_REPO, rel, target, repo_type="dataset", revision=revision)
        for doc in iter_jsonl(target):
            cid = str(doc.get("context_id"))
            if cid in wanted_cids:
                lookup[(cid, int(doc.get("rollout_k") or 0))] = {
                    "completion": doc.get("completion") or "",
                    "answer_aliases": doc.get("answer_aliases") or [],
                }
    return lookup


def s4_labels(
    picks_k: dict[str, int],
    packed_lookup: dict[tuple[str, int], dict],
    abstain_scores: dict[tuple[str, int], float | None],
) -> dict[str, str]:
    """Hallucination single-draw three-way labels (unit B's S4; no judge calls)."""
    fits = _fits2091()
    out = {}
    for cid, k in picks_k.items():
        row = packed_lookup.get((cid, k))
        if row is None:
            out[cid] = "missing_packed_row"
            continue
        out[cid] = fits.s4_single_draw_label(row, abstain_scores, context_id=cid, k=k)
    return out


# ── capture parity (MF-2) ─────────────────────────────────────────────────────
def capture_parity_full(greedy_views: dict[str, StoreView], banked: StoreView) -> dict:
    """Full-coverage cos(context_end_new, context_end_banked) per rung x layer."""
    out: dict[str, dict] = {}
    for job, view in greedy_views.items():
        cids = [c for c in view.context_ids() if c in banked.rows_by_ctx]
        per_layer = {}
        for layer in LAYERS:
            a = _unit_rows(view.vc(cids, layer).astype(np.float64))
            b = _unit_rows(banked.vc(cids, layer).astype(np.float64))
            per_layer[f"L{layer}"] = summarize(np.einsum("nd,nd->n", a, b))
        out[job] = {
            "n_overlap": len(cids),
            "n_new": len(view.context_ids()),
            "per_layer": per_layer,
        }
    return out


def capture_parity_probe(probe_view: StoreView, banked: StoreView) -> dict:
    """Deferred P2 probe cosines (plan §4.2 P2 either-or): probe rows vs banked rows."""
    out: dict[str, dict] = {}
    for kind in ("t1", "context_end"):
        per_layer = {}
        for layer in LAYERS:
            cos = []
            for i, m in enumerate(probe_view.meta):
                cid = str(m.get("context_id"))
                k = int(m.get("rollout_k") or 0)
                rows = banked.rows_by_ctx.get(cid)
                if not rows:
                    continue
                ks = {int(banked.meta[j].get("rollout_k") or 0): j for j in rows}
                if k not in ks:
                    continue
                a = probe_view.arrays[(kind, layer)][i].astype(np.float64)
                b = banked.arrays[(kind, layer)][ks[k]].astype(np.float64)
                den = np.linalg.norm(a) * np.linalg.norm(b)
                if den > 0:
                    cos.append(float(np.dot(a, b) / den))
            per_layer[f"L{layer}"] = summarize(np.array(cos))
        out[kind] = per_layer
    return out


# ── fits orchestration (R4) ───────────────────────────────────────────────────
def build_regime_targets(
    cids: list[str],
    banked: StoreView,
    greedy: StoreView,
    layer: int,
    picks: dict[str, dict] | None,
) -> dict[str, np.ndarray]:
    """(n, d) f32 answer-vector targets per regime for one layer (S1-shared picks)."""
    v5 = banked.t1_stack(cids, layer)
    if picks is None:
        k_of = {c: int(rng_for(f"vec-single::{c}").integers(K_ROLLOUTS)) for c in cids}
    else:
        k_of = {c: int(picks[c]["k"]) if c in picks else 0 for c in cids}
    return {
        "greedy": greedy.t1_first(cids, layer),
        "avg_k5": v5.mean(axis=1),
        "single": banked.t1_k(cids, layer, k_of),
    }


class FitTimer:
    """Plan §7 G3 fit-pilot gate: the FIRST fresh ``fit_pool_regime`` call is the pilot.

    ``observe(elapsed, report_dir, fid)`` records the first FRESH wrapper-call
    wall (resumed checkpoints never observe), extrapolates
    ``elapsed x FIT_TOTAL_CALLS / parallelism`` (parallelism 1 — the family
    driver is serial on VM CPU) against the §9 P4-fits booking
    (``FIT_WALL_BUDGET_S``), persists ``g3_fit_pilot_gate.json`` either way,
    and on a > 2x projection fires a DESIGNED halt (``rc=RC_FIT_WALL``,
    mirroring the ``probe_disk`` rc=8 shape) — never a silent descope: the
    registered remedy is the vectorize check (plan §7 G3;
    ``.claude/rules/vectorize-many-cell-fits.md``). The synthetic pilot path
    (``issue2091_fits.py --mode pilot-synthetic``) calls ``fit_pool_regime``
    directly and never routes through ``fit_setting_grids``, so this gate
    binds the production family run only. The halt fires AFTER the pilot
    fit's checkpoint is persisted, so a post-fix relaunch resumes past it.
    Under ``--phase all`` (one process) the module singleton observes once —
    the first fresh fit of family A (generic); under per-family ``--phase
    family`` invocations each process re-pilots on its own first fresh fit.
    """

    def __init__(self) -> None:
        self.elapsed_first: float | None = None

    def observe(self, elapsed: float, report_dir: Path, fid: str) -> None:
        if self.elapsed_first is not None:
            return
        self.elapsed_first = elapsed
        projected = elapsed * FIT_TOTAL_CALLS
        threshold = 2 * FIT_WALL_BUDGET_S
        verdict = "halt" if projected > threshold else "pass"
        report_path = report_dir / "g3_fit_pilot_gate.json"
        write_json_atomic(
            report_path,
            {
                "gate": "G3 fit pilot (plan §7; §9 P4-fits row)",
                "pilot_fit_id": fid,
                "measured_first_call_s": round(elapsed, 2),
                "total_calls": FIT_TOTAL_CALLS,
                "parallelism": 1,
                "projected_wall_s": round(projected, 1),
                "booked_wall_s": FIT_WALL_BUDGET_S,
                "threshold_s": threshold,
                "verdict": verdict,
                "remedy": "vectorize check per .claude/rules/vectorize-many-cell-fits.md",
            },
        )
        logger.info(
            "[fits] G3 pilot %s: %.1fs/call -> projected %.0fs over %d calls "
            "(threshold %.0fs) -> %s",
            fid,
            elapsed,
            projected,
            FIT_TOTAL_CALLS,
            threshold,
            verdict.upper(),
        )
        if verdict == "halt":
            print(
                f"[issue2091-p4] DESIGNED HALT rc={RC_FIT_WALL}: G3 fit pilot ({fid}) measured "
                f"{elapsed:.1f}s/call -> projected {projected:.0f}s over {FIT_TOTAL_CALLS} "
                f"wrapper calls > 2x the §9 P4-fits booking ({FIT_WALL_BUDGET_S:.0f}s). "
                f"Pilot fit checkpoint persisted; report at {report_path}. Do NOT descope: "
                f"run the vectorize check (.claude/rules/vectorize-many-cell-fits.md), "
                f"then relaunch (resumes past the persisted fit).",
                flush=True,
            )
            sys.exit(RC_FIT_WALL)


FIT_TIMER = FitTimer()


def fit_setting_grids(
    *,
    setting: str,
    pool_spec: dict,
    pool_x: np.ndarray,
    pool_y: dict[str, np.ndarray],
    eval_x: np.ndarray,
    eval_y: dict[str, np.ndarray],
    groups: np.ndarray,
    eval_groups: Sequence[str],
    fits_dir: Path,
    control: dict | None,
) -> dict:
    """One setting's R4 3x3 R^2 grid (+ identity/kNN diagnostics + control row).

    Per-fit persistence: the FitResult JSON (selection + diagnostics) is the
    checkpoint; on resume, predictions are REBUILT from the persisted
    selected-lambda per layer via one 1-element-grid core call (seconds) —
    no bulky prediction npz (the #813 per-item serialization lesson).
    """
    fits = _fits2091()
    result: dict = {"setting": setting, "pool": pool_spec, "fits": {}, "r2_grid": {}}
    inner_caches = None
    predictions: dict[str, np.ndarray] = {}
    for regime in REGIMES:
        fid = f"{setting}__{regime}"
        ck = fits_dir / f"{fid}.json"
        if ck.is_file():
            fr_json = json.loads(ck.read_text())
            lams = [pl["selection"]["best_lambda"] for pl in fr_json["per_layer"]]
            predictions[regime] = np.stack(
                [
                    fits.fit_predict_at_lambda(
                        pool_x[li].astype(np.float64),
                        pool_y[regime][li].astype(np.float64),
                        eval_x[li].astype(np.float64),
                        float(lams[li]),
                    )
                    for li in range(len(LAYERS))
                ]
            )
            result["fits"][regime] = fr_json
            logger.info("[fits] resume %s (re-predicted at persisted lambdas)", fid)
            continue
        if inner_caches is None:
            inner_caches = [
                fits.build_inner_caches(pool_x[li].astype(np.float64), groups)
                for li in range(len(LAYERS))
            ]
        t0 = time.time()
        fr = fits.fit_pool_regime(
            pool_x.astype(np.float64),
            pool_y[regime].astype(np.float64),
            eval_x.astype(np.float64),
            eval_y[regime].astype(np.float64),
            groups,
            layers=LAYERS,
            fit_id=fid,
            inner_caches=inner_caches,
            eval_groups=eval_groups,
        )
        elapsed = time.time() - t0
        write_json_atomic(ck, fr.to_json())
        result["fits"][regime] = fr.to_json()
        predictions[regime] = fr.predictions
        logger.info("[fits] unit %s elapsed=%.1fs", fid, elapsed)
        FIT_TIMER.observe(elapsed, fits_dir, fid)  # G3 gate: after the checkpoint write
    for li, layer in enumerate(LAYERS):
        result["r2_grid"][f"L{layer}"] = {
            fr_: {er: r2_score_rows(predictions[fr_][li], eval_y[er][li]) for er in REGIMES}
            for fr_ in REGIMES
        }
    if control is not None:
        ctrl = {}
        for regime in REGIMES:
            lam_per_layer = control["selected_lambda"][regime]
            preds = np.stack(
                [
                    fits.fit_predict_at_lambda(
                        control["x_pool"][li].astype(np.float64),
                        control["y_pool"][regime][li].astype(np.float64),
                        eval_x[li].astype(np.float64),
                        float(lam_per_layer[li]),
                    )
                    for li in range(len(LAYERS))
                ]
            )
            ctrl[regime] = {
                f"L{layer}": {er: r2_score_rows(preds[li], eval_y[er][li]) for er in REGIMES}
                for li, layer in enumerate(LAYERS)
            }
        result["control_r2"] = ctrl
        result["control_note"] = (
            "matched all-generic control: the family-A generic-core fit re-applied at its "
            "persisted selected lambdas to THIS setting's eval set (pool-composition control)"
        )
    result["_predictions"] = predictions  # in-memory handoff to behavioral readouts
    return result


def map963k_reference(maps_npz: Path, eval_x: np.ndarray, eval_y: dict[str, np.ndarray]) -> dict:
    """Frozen-963k reference row (f_u = 0 endpoint; #1739 map963k_reuse conventions).

    Apply = the COMMITTED loader + expression: ((x - x_mu) / x_sd) @ w + y_mu.
    The payload meta (fit_space / whitening provenance) is quoted for the R4
    caption (plan A16 — reference row only, never a matched arm).
    """
    from scripts.issue1739_map963k_readout import load_i1739_map

    out: dict = {"map_npz": str(maps_npz), "r2": {}, "meta": None}
    for li, layer in enumerate(LAYERS):
        w, x_mu, x_sd, y_mu, meta = load_i1739_map(maps_npz, layer)
        if out["meta"] is None and meta:
            out["meta"] = {k: meta.get(k) for k in ("fit_space", "whitening_provenance")}
        pred = ((eval_x[li].astype(np.float64) - x_mu) / x_sd) @ w + y_mu
        out["r2"][f"L{layer}"] = {er: r2_score_rows(pred, eval_y[er][li]) for er in REGIMES}
    return out


def behavioral_readouts(
    *,
    setting: str,
    layer_idx: int,
    layer: int,
    rb_vec: np.ndarray,
    eval_x: np.ndarray,
    eval_v5_mean: np.ndarray,
    predictions: dict[str, np.ndarray],
    dv_cols: dict[str, np.ndarray],
    labeled_pool_x: np.ndarray | None,
    labeled_pool_va: np.ndarray | None,
    labeled_pool_dv: dict[str, np.ndarray] | None,
    pool_groups: np.ndarray | None,
    dv_half_a: np.ndarray | None,
    dv_half_b: np.ndarray | None,
    boot: GroupBootstrap,
) -> dict:
    """R4-b: per-method-family Spearman rho per regime column at one layer.

    Method families (plan §4.2): pv_projection (label-free), supervised_context
    (S5 pool-side labeled rows only), map_pv_projection, map_supervised_answer,
    oracle_answer, disjoint_half. Returns ``_percontext_scores`` for the R3
    commonality decomposition (popped by the caller before persisting); the
    map families' per-context commonality scores are pinned to
    ``COMMONALITY_MAP_REGIME`` (recorded as ``commonality_regime`` in the
    persisted family blocks — never the dv_cols iteration order).
    """
    fits = _fits2091()
    percontext: dict[str, dict[str, np.ndarray]] = {}

    def rho_with_ci(score: np.ndarray, col: str) -> dict:
        dv = dv_cols[col]
        r = spearman(score, dv)
        mask = np.isfinite(dv) & np.isfinite(score)
        rs = rankdata_avg(
            np.nan_to_num(score, nan=np.nanmedian(score) if np.isfinite(score).any() else 0.0)
        )
        rd = rankdata_avg(np.nan_to_num(dv, nan=np.nanmedian(dv) if np.isfinite(dv).any() else 0.0))
        draws = boot.corr(rs, rd, mask=mask)
        return {
            "rho": None if not np.isfinite(r) else r,
            "n": int(mask.sum()),
            "ci95": GroupBootstrap.ci(draws),
        }

    out: dict = {}
    proj = eval_x[layer_idx].astype(np.float64) @ rb_vec
    percontext["pv_projection"] = {"score": proj}
    out["pv_projection"] = {c: rho_with_ci(proj, c) for c in dv_cols}

    if labeled_pool_x is not None and labeled_pool_dv is not None and pool_groups is not None:
        sup = {}
        y_dv = labeled_pool_dv["avg_k5"]
        keep = np.isfinite(y_dv)
        if keep.sum() >= 20:
            sel = fits.select_lambda(
                labeled_pool_x[layer_idx][keep].astype(np.float64),
                y_dv[keep, None],
                pool_groups[keep],
                where=f"{setting}/ctxread/L{layer}",
            )
            pred = fits.fit_predict_at_lambda(
                labeled_pool_x[layer_idx][keep].astype(np.float64),
                y_dv[keep, None],
                eval_x[layer_idx].astype(np.float64),
                sel.best_lambda,
            )[:, 0]
            percontext["supervised_context"] = {"score": pred}
            sup = {c: rho_with_ci(pred, c) for c in dv_cols}
            sup["selected_lambda"] = sel.best_lambda
            sup["selection"] = sel.to_json()  # full selector diagnostics (plan §8; n<d regime)
            sup["n_labeled_pool"] = int(keep.sum())
        else:
            sup = {"note": f"n_labeled_pool={int(keep.sum())} < 20 — readout skipped"}
        out["supervised_context"] = sup

        # answer-side readout (ridge banked-vA -> DV on POOL-side labeled rows; S5)
        if labeled_pool_va is not None and keep.sum() >= 20:
            sel_a = fits.select_lambda(
                labeled_pool_va[layer_idx][keep].astype(np.float64),
                y_dv[keep, None],
                pool_groups[keep],
                where=f"{setting}/ansread/L{layer}",
            )
            out["map_supervised_answer"] = {}
            out["oracle_answer"] = {}
            for col in dv_cols:
                regime = col if col in REGIMES else "avg_k5"
                vhat = predictions.get(regime)
                if vhat is not None:
                    score_map = fits.fit_predict_at_lambda(
                        labeled_pool_va[layer_idx][keep].astype(np.float64),
                        y_dv[keep, None],
                        vhat[layer_idx].astype(np.float64),
                        sel_a.best_lambda,
                    )[:, 0]
                    out["map_supervised_answer"][col] = rho_with_ci(score_map, col)
                    if regime == COMMONALITY_MAP_REGIME and "map_supervised_answer" not in (
                        percontext
                    ):
                        # R3 commonality score pinned to ONE regime (never dict-order)
                        percontext["map_supervised_answer"] = {"score": score_map}
                else:
                    out["map_supervised_answer"][col] = {"rho": None, "note": "no map prediction"}
            out["map_supervised_answer"]["commonality_regime"] = COMMONALITY_MAP_REGIME
            out["map_supervised_answer"]["selection"] = sel_a.to_json()
            score_oracle = fits.fit_predict_at_lambda(
                labeled_pool_va[layer_idx][keep].astype(np.float64),
                y_dv[keep, None],
                eval_v5_mean.astype(np.float64),
                sel_a.best_lambda,
            )[:, 0]
            percontext["oracle_answer"] = {"score": score_oracle}
            out["oracle_answer"] = {c: rho_with_ci(score_oracle, c) for c in dv_cols}
            out["oracle_answer"]["selection"] = sel_a.to_json()  # same ansread fit as map_*

    mp = {}
    for col in dv_cols:
        regime = col if col in REGIMES else "avg_k5"
        vhat = predictions.get(regime)
        if vhat is None:
            mp[col] = {"rho": None, "note": "no map prediction for regime"}
            continue
        score = vhat[layer_idx].astype(np.float64) @ rb_vec
        if regime == COMMONALITY_MAP_REGIME and "map_pv_projection" not in percontext:
            # R3 commonality score pinned to ONE regime (never dict-order)
            percontext["map_pv_projection"] = {"score": score}
        mp[col] = rho_with_ci(score, col)
    mp["commonality_regime"] = COMMONALITY_MAP_REGIME
    out["map_pv_projection"] = mp

    if dv_half_a is not None and dv_half_b is not None:
        r = spearman(dv_half_a, dv_half_b)
        out["disjoint_half"] = {
            "rho_half_vs_half": None if not np.isfinite(r) else r,
            "note": "DV noise reference: rollouts {0,2,4} mean vs {1,3} mean (2-vs-3 caveat, A2)",
        }
    out["_percontext_scores"] = percontext
    return out


def commonality_block(
    *,
    percontext_scores: dict[str, dict[str, np.ndarray]],
    dv_avg: np.ndarray,
    sigma_defs: dict[str, np.ndarray],
    p_arr: np.ndarray,
    boot: GroupBootstrap,
) -> dict:
    """R3-hero: per method family x sigma-def rank-error commonality + companion.

    Outcome = per-context |rank(method score) - rank(avg-K5 DV)| over the eval
    set; predictors = sigma_A (either definition) and P. Companion strip:
    unique_sigma - unique_P with a cluster-bootstrap CI (weighted correlation
    algebra on global ranks per draw).
    """
    out: dict = {}
    for sig_name, sig_arr in sigma_defs.items():
        sig_arr = np.asarray(sig_arr, dtype=np.float64)
        fam_out = {}
        for fam, pack in percontext_scores.items():
            score = pack.get("score")
            if score is None:
                continue
            mask = np.isfinite(score) & np.isfinite(dv_avg)
            r_s = rankdata_avg(np.nan_to_num(score, nan=0.0))
            r_d = rankdata_avg(np.nan_to_num(dv_avg, nan=0.0))
            rank_err = np.abs(r_s - r_d).astype(np.float64)
            rank_err[~mask] = np.nan
            cm = commonality(rank_err, sig_arr, p_arr)
            if cm.get("r2_full") is not None:
                m2 = mask & np.isfinite(sig_arr) & np.isfinite(p_arr)
                ru = rankdata_avg(np.nan_to_num(rank_err, nan=0.0))
                rs = rankdata_avg(np.nan_to_num(sig_arr, nan=0.0))
                rp = rankdata_avg(np.nan_to_num(p_arr, nan=0.0))
                r_y1 = boot.corr(ru, rs, mask=m2)
                r_y2 = boot.corr(ru, rp, mask=m2)
                r_12 = boot.corr(rs, rp, mask=m2)
                with np.errstate(invalid="ignore", divide="ignore"):
                    den = 1 - r_12**2
                    r2f = (r_y1**2 + r_y2**2 - 2 * r_y1 * r_y2 * r_12) / den
                    u_sig = r2f - r_y2**2
                    u_p = r2f - r_y1**2
                cm["companion_unique_sigma_minus_unique_p"] = {
                    "median": float(np.nanmedian(u_sig - u_p)),
                    "ci95": GroupBootstrap.ci(u_sig - u_p),
                }
            fam_out[fam] = cm
        out[sig_name] = fam_out
    return out


# ── phases ────────────────────────────────────────────────────────────────────
def phase_upload_judge_artifacts(args) -> None:
    """C/D duty (unit B handoff): rubric_parity + resolved_rubrics + pilot report -> HF.

    One bulk ``upload_folder`` commit to ``issue2091_decode/judge/`` — the
    realized-instrument record de-risks the issue-1739-worktree dependency
    unit B's rubric-parity smoke surfaced (its KEY FINDING).
    """
    hub = _hub()
    src = Path(args.out_root) / "greedy_dv"
    parity = src / "rubric_parity.json"
    if not parity.is_file():
        raise FileNotFoundError(
            f"{parity} missing — run issue2091_judge.py --phase rubric-smoke first"
        )
    with tempfile.TemporaryDirectory(prefix="i2091_judge_up_") as td:
        stagedir = Path(td) / "judge"
        stagedir.mkdir(parents=True)
        shutil.copy2(parity, stagedir / "rubric_parity.json")
        rr = src / "resolved_rubrics"
        if rr.is_dir():
            shutil.copytree(rr, stagedir / "resolved_rubrics")
        pilot = src / "pilot" / "gate_report.json"
        if pilot.is_file():
            (stagedir / "pilot").mkdir()
            shutil.copy2(pilot, stagedir / "pilot" / "gate_report.json")
        else:
            logger.warning("[upload-judge] pilot gate_report.json absent (pilot not yet run)")
        if args.skip_upload:
            logger.info("[upload-judge] SKIP (--skip-upload); staged tree at %s", stagedir)
            return
        url = hub._upload(stagedir, DATA_REPO, "dataset", f"{HF_PREFIX}/judge", raise_on_error=True)
        logger.info("[upload-judge] uploaded -> %s/judge (%s)", HF_PREFIX, url or "no-url")


def retained_dir(staging_root: Path) -> Path:
    d = staging_root / "retained"
    d.mkdir(parents=True, exist_ok=True)
    return d


def family_done_path(staging_root: Path, family: str) -> Path:
    return retained_dir(staging_root) / f"family_{family}_done.json"


class BatteryTimer:
    """Plan §9 battery-row gate: timed B=50 sub-battery before the full B.

    ``resolve(run_pilot)`` times ONE production-code battery pass at B=50 and
    extrapolates; over 2x the 0.5 h booking fires the pre-registered descope
    lever B -> 500 (recorded in the outputs).
    """

    def __init__(self, b_requested: int) -> None:
        self.b_requested = b_requested
        self.decided_b: int | None = None
        self.elapsed_pilot: float | None = None

    def resolve(self, run_pilot) -> int:
        if self.decided_b is not None:
            return self.decided_b
        if self.b_requested <= B_BOOT_PILOT:
            self.decided_b = self.b_requested
            return self.decided_b
        t0 = time.time()
        run_pilot(B_BOOT_PILOT)
        self.elapsed_pilot = time.time() - t0
        projected = self.elapsed_pilot * (self.b_requested / B_BOOT_PILOT)
        self.decided_b = B_BOOT_DESCOPED if projected > BATTERY_WALL_BUDGET_S else self.b_requested
        logger.info(
            "[battery] pilot B=%d took %.1fs -> projected %.0fs (budget %.0fs) -> B=%d%s",
            B_BOOT_PILOT,
            self.elapsed_pilot,
            projected,
            BATTERY_WALL_BUDGET_S,
            self.decided_b,
            " (DESCOPE LEVER FIRED)" if self.decided_b == B_BOOT_DESCOPED else "",
        )
        return self.decided_b


def setting_stat_block(
    *,
    setting: str,
    behavior: str,
    cids: list[str],
    banked: StoreView,
    greedy: StoreView | None,
    scores: np.ndarray | None,
    score_ids: list[str] | None,
    groups: list[str],
    groups2: list[str] | None,
    b_boot: int,
    rb_vec_by_layer: dict[int, np.ndarray] | None,
) -> dict:
    """R1/R2/R3(moderators+ceilings+guardrails)/R5 per-setting statistics."""
    block: dict = {"setting": setting, "behavior": behavior, "n_contexts": len(cids)}
    boot = GroupBootstrap(groups, b_boot, f"{setting}", groups2=groups2)
    boot_prefix_only = (
        GroupBootstrap(groups, b_boot, f"{setting}::prefix-only") if groups2 is not None else None
    )

    # R1 dispersion (per layer; headline L19)
    r1 = {}
    v5_by_layer = {}
    for layer in LAYERS:
        v5 = banked.t1_stack(cids, layer)
        v5_by_layer[layer] = v5
        disp = pairwise_cos_dispersion(v5)
        r1[f"L{layer}"] = {
            "percontext": {"context_id": cids, "dispersion": disp},
            "summary": summarize(disp),
            "boot_ci_median": GroupBootstrap.ci(boot.median(disp)),
        }
    block["r1"] = r1

    # R2 Delta (needs the greedy vectors)
    if greedy is not None:
        r2 = {}
        for layer in LAYERS:
            g = greedy.t1_first(cids, layer).astype(np.float64)
            v5 = v5_by_layer[layer].astype(np.float64)
            d = delta_ctx(g, v5)
            ranks = exchangeability_ranks(g, v5)
            disp = pairwise_cos_dispersion(v5)
            q = np.ceil(rankdata_avg(disp) / len(cids) * 5).clip(1, 5).astype(int)
            half = disjoint_half_agreement(v5, f"{setting}::L{layer}")
            med_draws = boot.median(d["delta"])
            entry = {
                "percontext": {
                    "context_id": cids,
                    "delta": d["delta"],
                    "cos_g_kmean": d["cos_g_kmean"],
                    "dispersion_quintile": q,
                },
                "median_delta": float(np.median(d["delta"])),
                "boot_ci_median": GroupBootstrap.ci(med_draws),
                "severe_tail_rate": float((d["delta"] < SEVERE_TAIL).mean()),
                "common_language_p": float((d["delta"] > 0).mean()),
                "jackknife": jackknife_delta_band(g, v5),
                "exchangeability": {
                    "mean_rank": float(ranks.mean()),
                    "expected_mean": (K_ROLLOUTS + 2) / 2.0,
                    "rank_hist": {str(r): int((ranks == r).sum()) for r in np.unique(ranks)},
                    "boot_ci_mean_rank": GroupBootstrap.ci(boot.mean(ranks)),
                },
                "quintile_curve": {
                    "quintile": list(range(1, 6)),
                    "cos_g_kmean_median": [
                        float(np.median(d["cos_g_kmean"][q == i])) if (q == i).any() else None
                        for i in range(1, 6)
                    ],
                    "disjoint_half_median": [
                        float(np.median(half[q == i])) if (q == i).any() else None
                        for i in range(1, 6)
                    ],
                    "note": "disjoint-half at K=5 is 2-vs-3: approximately noise-matched (A2)",
                },
            }
            if boot_prefix_only is not None:
                entry["boot_ci_median_prefix_only"] = GroupBootstrap.ci(
                    boot_prefix_only.median(d["delta"])
                )
                entry["clustering_note"] = (
                    "primary CI = two-way (prefix group_key x meta.question_key, A17 "
                    "fallback-in-stronger-form); prefix-only variant reported alongside"
                )
            r2[f"L{layer}"] = entry
        block["r2"] = r2

    # Score-side legs (R3 inputs/ceilings/guardrails + R5)
    if scores is not None and score_ids:
        sid_to_row = {c: i for i, c in enumerate(score_ids)}
        keep = [c for c in cids if c in sid_to_row]
        if keep:
            sc = scores[np.array([sid_to_row[c] for c in keep], dtype=int)]
            cid_pos = {c: i for i, c in enumerate(cids)}
            boot_sc = GroupBootstrap(
                [groups[cid_pos[c]] for c in keep], b_boot, f"{setting}::scores"
            )
            pol = polarization_stats(sc, keep)
            g_pol_ci = None
            if pol["n_middling"] >= 3:
                g_draws = boot_sc.mean(pol["f_mid"], mask=pol["middling_mask"]) - 0.5
                g_pol_ci = GroupBootstrap.ci(g_draws)
            floor_share = float(np.nanmean(pol["mu"] <= 1.0)) if len(keep) else None
            uninformative = bool(
                behavior == "evil" and (pol["n_middling"] < 20 or (floor_share or 0.0) > 0.9)
            )
            block["r5"] = {
                "percontext": {
                    "context_id": keep,
                    "mu": pol["mu"],
                    "sd": pol["sd"],
                    "p": pol["p"],
                    "f_mid": pol["f_mid"],
                },
                "n_middling": pol["n_middling"],
                "mean_f_mid": pol["mean_f_mid"],
                "q_pol": pol["q_pol"],
                "g_pol": {"value": pol["g_pol"], "ci95": g_pol_ci},
                "floor_share_mu_le_1": floor_share,
                "uninformative": uninformative,
                "lattice_note": (
                    "K=5 lattice: only discrete (mu, sd) pairs attainable; "
                    "population-level claims only (body Result 5)"
                ),
            }
            # R3 moderators + ceilings + guardrails
            sigma_defs = {}
            v5h = v5_by_layer[HEADLINE_LAYER][np.array([cid_pos[c] for c in keep], dtype=int)]
            sig_tot = np.sqrt(
                np.mean(
                    np.linalg.norm(
                        v5h.astype(np.float64) - v5h.astype(np.float64).mean(axis=1, keepdims=True),
                        axis=2,
                    )
                    ** 2,
                    axis=1,
                )
            )
            sigma_defs["sigma_a_total"] = sig_tot
            if rb_vec_by_layer is not None:
                projk = np.einsum(
                    "nkd,d->nk", v5h.astype(np.float64), rb_vec_by_layer[HEADLINE_LAYER]
                )
                sigma_defs["sigma_a_proj"] = projk.std(axis=1, ddof=0)
            jdv = judge_draw_variance(behavior)
            guard = {
                f"rho_{name}_vs_p": spearman(arr, pol["p"]) for name, arr in sigma_defs.items()
            }
            pa = polarization_stats(sc[:, [0, 2, 4]], keep)
            pb = polarization_stats(sc[:, [1, 3]], keep)
            sig_a = pairwise_cos_dispersion(v5h[:, [0, 2, 4], :])
            sig_b_pair = pairwise_cos_dispersion(v5h[:, [1, 3], :])
            block["r3_ceilings"] = column_ceilings(sc, jdv)
            block["r3_guardrails"] = {
                **guard,
                "split_half": {
                    "p": split_half_reliability(pa["p"], pb["p"]),
                    "sigma_a_dispersion": split_half_reliability(sig_a, sig_b_pair),
                },
                "note": "analyzer-weighed diagnostics, not kill gates (plan §6 R3 guardrails)",
            }
            block["r3_inputs"] = {
                "context_id": keep,
                "sigma_defs": {k: v for k, v in sigma_defs.items()},
                "p": pol["p"],
                "scores_within_sd": pol["sd"],
                "scores_mu": pol["mu"],
            }
    return block


def hallu_own_rung_r5(rows: list[dict], cids: list[str]) -> dict | None:
    """R5 for hallucination OWN rungs from the banked 3-way construct (plan A2).

    Per-rollout labels are binary (fabricated vs not, over decided draws), so
    P == 1 by construction wherever 0 < dv < 1 — DEFINITIONAL, not a finding.
    mu = 100*dv; sd = 100*sqrt(dv(1-dv)); f_mid = 0 (binary draws never sit
    in [25, 75]).
    """
    by_ctx = {str(r["context_id"]): r for r in rows}
    keep = [c for c in cids if by_ctx.get(c, {}).get("dv") is not None]
    if not keep:
        return None
    dv = np.array([float(by_ctx[c]["dv"]) for c in keep])
    mu = 100.0 * dv
    sd = 100.0 * np.sqrt(np.clip(dv * (1.0 - dv), 0.0, None))
    middling = (mu >= MID_LO) & (mu <= MID_HI)
    return {
        "percontext": {
            "context_id": keep,
            "mu": mu,
            "sd": sd,
            "p": np.where((dv > 0) & (dv < 1), 1.0, np.nan),
        },
        "n_middling": int(middling.sum()),
        "mean_f_mid": 0.0 if middling.any() else None,
        "q_pol": -0.4 if middling.any() else None,
        "g_pol": {"value": -0.5 if middling.any() else None, "ci95": None},
        "definitional": True,
        "lattice_note": (
            "own-rung 3-way construct: per-rollout labels are binary, so P == 1 by "
            "construction — definitional, never plotted with the 0-100 panels (plan A2)"
        ),
    }


# ── banked-only smoke (real-data leg; brief requirement) ─────────────────────
def phase_banked_smoke(args) -> int:
    """R5 + R3-plot-1 statistics on committed #1739 banked labeling.json (no staging).

    Zero-risk real-data leg: one behavior, tiny B, scratch out-root. Exercises
    the production statistics code (score matrix -> variance components /
    ceilings / polarization / cluster bootstrap) end-to-end on real banked rows.
    """
    behavior = args.behavior or "sycophancy"
    rows = [r for r in load_banked_dv(behavior) if r.get("per_rollout_scores")]
    if args.limit:
        rows = rows[: args.limit]
    scores, ids = per_rollout_score_matrix(rows)
    groups = [str(r.get("group_key") or r["context_id"]) for r in rows]
    jdv = judge_draw_variance(behavior)
    vc = variance_components(scores)
    ceils = column_ceilings(scores, jdv)
    pol = polarization_stats(scores, ids)
    boot = GroupBootstrap(groups, args.boot_b, f"banked-smoke::{behavior}")
    g_ci = (
        GroupBootstrap.ci(boot.mean(pol["f_mid"], mask=pol["middling_mask"]) - 0.5)
        if pol["n_middling"] >= 3
        else None
    )
    out = {
        "phase": "banked-smoke",
        "behavior": behavior,
        "n_rows": len(rows),
        "boot_b": args.boot_b,
        "judge_draw_var": jdv,
        "r3_plot1": {
            "variance_components": vc,
            "ceilings": {k: ceils[k] for k in ("ceil_greedy", "ceil_avg_k5", "ceil_single")},
        },
        "r5": {
            "n_middling": pol["n_middling"],
            "mean_f_mid": pol["mean_f_mid"],
            "q_pol": pol["q_pol"],
            "g_pol": {"value": pol["g_pol"], "ci95": g_ci},
            "mu_median": float(np.nanmedian(pol["mu"])),
            "p_median": float(np.nanmedian(pol["p"])),
        },
        "meta": {**_provenance_meta(), "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
    }
    dest = Path(args.out_root) / f"banked_smoke_{behavior}.json"
    write_json_atomic(dest, out)
    icc = vc["reliability_kmean"]
    print(
        f"[banked-smoke] {behavior}: n={len(rows)} icc_kmean={icc if icc is None else round(icc, 4)} "
        f"ceil_avg={ceils['ceil_avg_k5'] and round(ceils['ceil_avg_k5'], 4)} "
        f"n_middling={pol['n_middling']} "
        f"mean_f_mid={pol['mean_f_mid'] and round(pol['mean_f_mid'], 4)} -> {dest}",
        flush=True,
    )
    return 0


# ── family drivers ────────────────────────────────────────────────────────────
def _cx_rows(cx_last, idx_list: list[int], layer: int) -> np.ndarray:
    """(n, d) f32 rows of the LMSYS bundle context tensor at one layer."""
    import torch

    return cx_last[torch.as_tensor(idx_list), layer, :].to(torch.float32).numpy()


def _seeded_ids(ids: list[str], k: int, tag: str) -> list[str]:
    ids = sorted(dict.fromkeys(ids))
    if len(ids) < k:
        raise ValueError(f"{tag}: need {k}, have {len(ids)}")
    rng = rng_for(f"seedsel::{tag}")
    idx = sorted(rng.choice(len(ids), size=k, replace=False).tolist())
    return [ids[i] for i in idx]


def _seeded_idx(idx_list: list[int], k: int, tag: str) -> list[int]:
    if len(idx_list) < k:
        raise ValueError(f"{tag}: need {k}, have {len(idx_list)}")
    rng = rng_for(f"seedsel::{tag}")
    pick = sorted(rng.choice(len(idx_list), size=k, replace=False).tolist())
    return [idx_list[i] for i in pick]


def _lmsys_targets(v_greedy, v_stoch, sub_idx, single_pick, rows: np.ndarray, layer: int) -> dict:
    vs = v_stoch[layer][rows]  # (n, 10, d)
    sel = sub_idx[rows]  # (n, 5)
    take = vs[np.arange(len(rows))[:, None], sel, :]
    return {
        "greedy": v_greedy[layer][rows],
        "avg_k5": take.mean(axis=1).astype(np.float32),
        "single": vs[np.arange(len(rows)), single_pick[rows], :],
    }


def s1_picks_from_wcrung(staging: Path, revision: str, cids: list[str]) -> dict[str, dict]:
    """S1 picks for wildchat contexts (index shared between DV + vector draws).

    A context judged under several rubrics keeps the FIRST behavior's pick;
    a context with no judged scores gets a seeded vector-side draw (excluded
    from every single-draw DV column, per S1). Cross-behavior divergence: the
    vector-side targets use these first-behavior-precedence picks while each
    behavior's single-draw DV column uses its OWN ``s1_picks`` — the indices
    can diverge on contexts whose per-behavior all-None drop patterns differ
    (rare; identical kept-sets yield identical picks via the shared
    context-keyed rng). Recorded as ``s1_note`` in the generic r4 grid.
    """
    picks: dict[str, dict] = {}
    for behavior in BEHAVIORS:
        rows = load_wcrung_dv(behavior, staging, revision)
        for cid, pick in s1_picks([r for r in rows if r.get("per_rollout_scores")]).items():
            picks.setdefault(cid, pick)
    out = {}
    for c in cids:
        out[c] = picks.get(c) or {
            "k": int(rng_for(f"wc-single::{c}").integers(K_ROLLOUTS)),
            "dv": None,
            "dv_included": False,
        }
    return out


def _dump_partial(staging: Path, family: str, obj: dict) -> None:
    p = retained_dir(staging) / f"partial_{family}.json"
    write_json_atomic(p, json.loads(json.dumps(obj, default=_json_default)))
    logger.info("[partial] wrote %s (%.1f MB)", p, p.stat().st_size / 1e6)


def run_family_generic(args) -> None:
    """Family A: generic-core reduce + generic/LMSYS legs, then reap (MF-1 step 1)."""
    import torch

    staging = Path(args.staging_root)
    done = family_done_path(staging, "generic")
    if done.is_file() and not args.force:
        logger.info("[family:generic] done sentinel present — skip (%s)", done)
        return
    assert_family_headroom(staging, "generic")
    revision = args.dataset_revision
    contexts_dir = stage_contexts_tree(staging, revision)
    ret = retained_dir(staging)
    fits_dir = ret / "fits"
    fits_dir.mkdir(exist_ok=True)
    fits = _fits2091()

    # ── stage inputs ──
    keep_store = store_basename_keep(("t1", "context_end"), LAYERS)
    wcrung_dir = staging / "wcrung_store"
    stage_prefix_files(WCRUNG_STORE_PREFIX, wcrung_dir, revision=revision, keep=keep_store)
    greedy_wc_dir = staging / "greedy_wildchat"
    stage_prefix_files(
        f"{HF_PREFIX}/capture_store/greedy_wildchat",
        greedy_wc_dir,
        revision=revision,
        keep=keep_store,
    )
    vstore_dir = staging / "v_store"
    stage_prefix_files(
        V_STORE_PREFIX, vstore_dir, revision=V_STORE_REVISION, keep=lambda n: n.endswith(".pt")
    )
    bundle_path = staging / "lmsys_bundle" / "train_context_vectors.pt"
    if not bundle_path.is_file():
        _hub().stage_hub_file(DATA_REPO, LMSYS_BUNDLE_PATH, bundle_path, repo_type="dataset")
    maps_path = ret / "maps" / "context_end__ufull.npz"
    if not maps_path.is_file():
        _hub().stage_hub_file(DATA_REPO, MAP963K_NPZ_PATH, maps_path, repo_type="dataset")

    # ── wildchat rung objects ──
    wc_rows = load_job_contexts(contexts_dir, "wildchat")
    wc_pool = [str(r["context_id"]) for r in wc_rows if r["split"] == "pool"]
    wc_eval = [str(r["context_id"]) for r in wc_rows if r["split"] == "eval"]
    if args.limit:
        wc_pool, wc_eval = wc_pool[: args.limit], wc_eval[: args.limit]
    banked_wc = StoreView(wcrung_dir)
    greedy_wc = StoreView(greedy_wc_dir)

    # ── LMSYS objects (bundle + v_store) ──
    common = _common1073()
    cap = _cap1073()
    bundle = common.load_bundle(bundle_path, expected_layers=28, expected_hidden=3584, min_n=4900)
    n_lmsys = bundle["cx_last"].shape[0]
    cx_last = bundle["cx_last"]
    from scripts.issue1073_neardup_sensitivity import neardup_cluster_ids

    cluster_of_row, _diag = neardup_cluster_ids(list(bundle["prompts"]))
    lmsys_groups_all = [f"nd{int(c)}" for c in cluster_of_row]
    rng = rng_for("lmsys-split")
    clusters = np.unique(cluster_of_row)
    perm = rng.permutation(len(clusters))
    pool_clusters = set(clusters[perm[: len(clusters) // 2]].tolist())
    lmsys_pool_idx = [i for i in range(n_lmsys) if cluster_of_row[i] in pool_clusters]
    lmsys_eval_idx = [i for i in range(n_lmsys) if cluster_of_row[i] not in pool_clusters]

    v_greedy = {ly: np.zeros((n_lmsys, 3584), dtype=np.float32) for ly in LAYERS}
    v_stoch = {ly: np.zeros((n_lmsys, LMSYS_K_BANKED, 3584), dtype=np.float32) for ly in LAYERS}
    seen_g = np.zeros(n_lmsys, dtype=bool)
    seen_s = np.zeros((n_lmsys, LMSYS_K_BANKED), dtype=bool)
    span_lens = np.zeros(n_lmsys, dtype=np.int64)
    for arm in ("greedy", "stoch10"):
        for _p, shard in cap.iter_shards(vstore_dir, arm):
            layer_pos = {int(ly): int(i) for i, ly in enumerate(list(shard["layers"]))}
            summ = shard["summ"]
            sp = shard.get("span_lens")
            for row, (ci, ri) in enumerate(list(shard["index"])):
                ci, ri = int(ci), int(ri)
                for ly in LAYERS:
                    vec = summ[row, layer_pos[ly], :].to(torch.float32).numpy()
                    if arm == "greedy":
                        v_greedy[ly][ci] = vec
                    else:
                        v_stoch[ly][ci, ri] = vec
                if arm == "greedy":
                    seen_g[ci] = True
                    if sp is not None:
                        span_lens[ci] = int(sp[row])
                else:
                    seen_s[ci, ri] = True
    if not seen_g.all() or not seen_s.all():
        raise RuntimeError(
            f"v_store coverage incomplete: greedy {int(seen_g.sum())}/{n_lmsys}, "
            f"stoch {int(seen_s.sum())}/{n_lmsys * LMSYS_K_BANKED}"
        )
    sub_rng = rng_for("lmsys-5of10")
    sub_idx = np.stack([sub_rng.permutation(LMSYS_K_BANKED)[:K_ROLLOUTS] for _ in range(n_lmsys)])
    single_pick = sub_idx[:, 0]

    # ── generic-core pool matrices (plan §4.2 pools) ──
    n_wc_core = min(fits.GENERIC_WC, len(wc_pool))
    n_lm_core = min(fits.GENERIC_LMSYS, len(lmsys_pool_idx))
    if (n_wc_core, n_lm_core) != (fits.GENERIC_WC, fits.GENERIC_LMSYS):
        logger.warning(
            "[family:generic] core below registered sizes (wc %d/%d, lmsys %d/%d) — smoke slice",
            n_wc_core,
            fits.GENERIC_WC,
            n_lm_core,
            fits.GENERIC_LMSYS,
        )
    wc_pool_sel = _seeded_ids(wc_pool, n_wc_core, "generic-wc")
    lmsys_pool_sel = _seeded_idx(lmsys_pool_idx, n_lm_core, "generic-lmsys")
    core_ids = [f"lmsys::{i}" for i in lmsys_pool_sel] + wc_pool_sel
    core_groups = np.array(
        [lmsys_groups_all[i] for i in lmsys_pool_sel] + wc_pool_sel, dtype=object
    )  # wcrung group_key == context (design effect 1.0, plan §6)
    wc_picks = s1_picks_from_wcrung(staging, revision, wc_pool_sel + wc_eval)
    x_core = np.stack(
        [
            np.concatenate([_cx_rows(cx_last, lmsys_pool_sel, ly), banked_wc.vc(wc_pool_sel, ly)])
            for ly in LAYERS
        ]
    )
    y_core: dict[str, np.ndarray] = {}
    for regime in REGIMES:
        per_layer = []
        for ly in LAYERS:
            lm = _lmsys_targets(
                v_greedy, v_stoch, sub_idx, single_pick, np.array(lmsys_pool_sel), ly
            )[regime]
            wc = build_regime_targets(wc_pool_sel, banked_wc, greedy_wc, ly, wc_picks)[regime]
            per_layer.append(np.concatenate([lm, wc]))
        y_core[regime] = np.stack(per_layer)
    np.savez(  # plain savez, never savez_compressed (#813)
        ret / "generic_core.tmp.npz",
        x=x_core,
        **{f"y_{r}": y_core[r] for r in REGIMES},
        groups=core_groups.astype(str),
        ids=np.array(core_ids, dtype=str),
    )
    os.replace(ret / "generic_core.tmp.npz", ret / "generic_core.npz")

    # ── generic eval matrices + generic-panel fits ──
    x_eval = np.stack(
        [
            np.concatenate([_cx_rows(cx_last, lmsys_eval_idx, ly), banked_wc.vc(wc_eval, ly)])
            for ly in LAYERS
        ]
    )
    y_eval: dict[str, np.ndarray] = {}
    for regime in REGIMES:
        per_layer = []
        for ly in LAYERS:
            lm = _lmsys_targets(
                v_greedy, v_stoch, sub_idx, single_pick, np.array(lmsys_eval_idx), ly
            )[regime]
            wc = build_regime_targets(wc_eval, banked_wc, greedy_wc, ly, wc_picks)[regime]
            per_layer.append(np.concatenate([lm, wc]))
        y_eval[regime] = np.stack(per_layer)
    eval_groups = [lmsys_groups_all[i] for i in lmsys_eval_idx] + wc_eval

    pool_spec = {
        "pool_tag": "generic",
        "n_admix": 0,
        "f_u_realized": 0.0,
        "f_u_target": fits.TARGET_F_U,
        "u_pool": len(core_ids),
    }
    grid = fit_setting_grids(
        setting="generic",
        pool_spec=pool_spec,
        pool_x=x_core,
        pool_y=y_core,
        eval_x=x_eval,
        eval_y=y_eval,
        groups=core_groups,
        eval_groups=eval_groups,
        fits_dir=fits_dir,
        control=None,
    )
    grid["map963k"] = map963k_reference(maps_path, x_eval, y_eval)
    control_lams = {
        regime: [pl["selection"]["best_lambda"] for pl in grid["fits"][regime]["per_layer"]]
        for regime in REGIMES
    }
    write_json_atomic(ret / "generic_control_lambdas.json", control_lams)
    preds = grid.pop("_predictions")

    # ── behavioral rho on the wildchat eval half (all three rubrics; R4-b) ──
    rb_arr, rb_names = _store_io().load_rb_bank()
    b_boot = args.boot_b
    wc_eval_off = len(lmsys_eval_idx)
    wc_boot = GroupBootstrap(wc_eval, b_boot, "generic::wc-rho")
    wc_dv_banked = {b: load_wcrung_dv(b, staging, revision) for b in BEHAVIORS}
    behavioral = {}
    wc_commonality = {}
    for behavior in BEHAVIORS:
        rb_vec = rb_arr[HEADLINE_LAYER, rb_names.index(behavior), :]
        rows_b = [r for r in wc_dv_banked[behavior] if r.get("per_rollout_scores")]
        sm, sids = per_rollout_score_matrix(rows_b)
        sidx = {c: i for i, c in enumerate(sids)}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            avg_dv = {
                c: float(np.nanmean(sm[i])) for c, i in sidx.items() if np.isfinite(sm[i]).any()
            }
        picks = s1_picks(rows_b)
        try:
            gdv_payload = load_greedy_dv(Path(args.out_root), behavior)
            greedy_dv = greedy_dv_by_ctx(gdv_payload, wildchat_graded=(behavior == "hallucination"))
        except FileNotFoundError:
            greedy_dv = {}
            logger.warning(
                "[family:generic] greedy DV absent for %s — greedy rho column skipped", behavior
            )
        dv_cols = {
            "avg_k5": np.array([avg_dv.get(c, np.nan) for c in wc_eval]),
            "single": np.array(
                [
                    picks[c]["dv"] if c in picks and picks[c]["dv"] is not None else np.nan
                    for c in wc_eval
                ]
            ),
        }
        if greedy_dv:
            dv_cols["greedy"] = np.array([greedy_dv.get(c, np.nan) for c in wc_eval])
        li = LAYERS.index(HEADLINE_LAYER)
        wc_x_eval = x_eval[:, wc_eval_off:, :]
        wc_preds = {r: p[:, wc_eval_off:, :] for r, p in preds.items()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            half_a = np.array(
                [np.nanmean(sm[sidx[c], [0, 2, 4]]) if c in sidx else np.nan for c in wc_eval]
            )
            half_b = np.array(
                [np.nanmean(sm[sidx[c], [1, 3]]) if c in sidx else np.nan for c in wc_eval]
            )
        ro = behavioral_readouts(
            setting=f"generic-wildchat::{behavior}",
            layer_idx=li,
            layer=HEADLINE_LAYER,
            rb_vec=rb_vec,
            eval_x=wc_x_eval,
            eval_v5_mean=banked_wc.t1_stack(wc_eval, HEADLINE_LAYER).mean(axis=1),
            predictions=wc_preds,
            dv_cols=dv_cols,
            labeled_pool_x=np.stack([banked_wc.vc(wc_pool_sel, ly) for ly in LAYERS]),
            labeled_pool_va=np.stack(
                [banked_wc.t1_stack(wc_pool_sel, ly).mean(axis=1) for ly in LAYERS]
            ),
            labeled_pool_dv={"avg_k5": np.array([avg_dv.get(c, np.nan) for c in wc_pool_sel])},
            pool_groups=np.array(wc_pool_sel, dtype=object),
            dv_half_a=half_a,
            dv_half_b=half_b,
            boot=wc_boot,
        )
        pcs = ro.pop("_percontext_scores")
        behavioral[behavior] = ro
        # commonality over the wildchat eval half (R3 wildchat panel)
        v5e = banked_wc.t1_stack(wc_eval, HEADLINE_LAYER)
        sig_tot = np.sqrt(
            np.mean(
                np.linalg.norm(
                    v5e.astype(np.float64) - v5e.astype(np.float64).mean(axis=1, keepdims=True),
                    axis=2,
                )
                ** 2,
                axis=1,
            )
        )
        projk = np.einsum("nkd,d->nk", v5e.astype(np.float64), rb_vec)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mu_e = np.array([avg_dv.get(c, np.nan) for c in wc_eval])
            sd_e = np.array(
                [np.sqrt(np.nanvar(sm[sidx[c]], ddof=0)) if c in sidx else np.nan for c in wc_eval]
            )
        with np.errstate(invalid="ignore", divide="ignore"):
            ceil_e = np.sqrt(np.clip(mu_e * (100 - mu_e), 0, None))
            p_e = np.where(ceil_e > 0, sd_e / ceil_e, np.nan)
        wc_commonality[behavior] = commonality_block(
            percontext_scores=pcs,
            dv_avg=dv_cols["avg_k5"],
            sigma_defs={"sigma_a_total": sig_tot, "sigma_a_proj": projk.std(axis=1, ddof=0)},
            p_arr=p_e,
            boot=wc_boot,
        )
    grid["behavioral_rho_wildchat_L19"] = behavioral
    grid["s1_note"] = (
        "S1 shared draw index (wildchat rung): vector-side single/avg_k5 targets use "
        "first-behavior-precedence picks (s1_picks_from_wcrung); each behavior's "
        "single-draw DV column uses its own picks — indices can diverge on contexts "
        "whose per-behavior all-None drop patterns differ (rare; identical kept-sets "
        "yield identical picks via the shared context-keyed rng)"
    )

    # ── stat blocks: wildchat + lmsys settings ──
    wc_all = wc_pool_sel + wc_eval
    blocks = {}
    battery = BatteryTimer(b_boot)

    def _run_wc(bb: int):
        return setting_stat_block(
            setting="wildchat",
            behavior="generic",
            cids=wc_all,
            banked=banked_wc,
            greedy=greedy_wc,
            scores=None,
            score_ids=None,
            groups=wc_all,
            groups2=None,
            b_boot=bb,
            rb_vec_by_layer=None,
        )

    b_eff = battery.resolve(_run_wc)
    blocks["wildchat"] = _run_wc(b_eff)
    wc_score_blocks = {}
    for behavior in BEHAVIORS:
        rows_b = [r for r in wc_dv_banked[behavior] if r.get("per_rollout_scores")]
        sm, sids = per_rollout_score_matrix(rows_b)
        rb_vec_by_layer = {ly: rb_arr[ly, rb_names.index(behavior), :] for ly in LAYERS}
        cids_b = [c for c in wc_all if c in set(sids)]
        blk = setting_stat_block(
            setting=f"wildchat::{behavior}",
            behavior=behavior,
            cids=cids_b,
            banked=banked_wc,
            greedy=greedy_wc,
            scores=sm,
            score_ids=sids,
            groups=cids_b,
            groups2=None,
            b_boot=b_eff,
            rb_vec_by_layer=rb_vec_by_layer,
        )
        blk["r3_commonality"] = wc_commonality.get(behavior)
        wc_score_blocks[behavior] = blk

    lmsys_all = np.array(lmsys_pool_sel + lmsys_eval_idx)
    lm_blocks = {}
    for layer in LAYERS:
        vs = v_stoch[layer][lmsys_all]
        v5 = vs[np.arange(len(lmsys_all))[:, None], sub_idx[lmsys_all], :]
        g = v_greedy[layer][lmsys_all]
        disp = pairwise_cos_dispersion(v5)
        d = delta_ctx(g.astype(np.float64), v5.astype(np.float64))
        lm_boot = GroupBootstrap(
            [lmsys_groups_all[i] for i in lmsys_all], b_eff, f"lmsys::L{layer}"
        )
        lm_blocks[f"L{layer}"] = {
            "dispersion_summary": summarize(disp),
            "dispersion_boot_ci_median": GroupBootstrap.ci(lm_boot.median(disp)),
            "median_delta": float(np.median(d["delta"])),
            "delta_boot_ci_median": GroupBootstrap.ci(lm_boot.median(d["delta"])),
            "severe_tail_rate": float((d["delta"] < SEVERE_TAIL).mean()),
            "percontext": {"idx": lmsys_all, "delta": d["delta"], "dispersion": disp},
            "median_answer_len": float(np.median(span_lens[lmsys_all])),
            "deviations_note": (
                "banked-LMSYS same-campaign cell: K 5-of-10 subsample; shown for context, "
                "never the headline (plan A1); rig-drift triangulation cell (MF-2 weigh rule)"
            ),
        }
    blocks["lmsys"] = {"per_layer": lm_blocks, "n_contexts": int(len(lmsys_all))}

    partial = {
        "blocks": blocks,
        "wc_score_blocks": wc_score_blocks,
        "r4_generic": grid,
        "battery_b_effective": b_eff,
        "battery_pilot_s": battery.elapsed_pilot,
    }
    _dump_partial(staging, "generic", partial)

    # ── reap family-A inputs (MF-1) ──
    for path in (vstore_dir, staging / "lmsys_bundle", wcrung_dir, greedy_wc_dir):
        reap(path)
    write_json_atomic(done, {"ts": time.time(), "b_effective": b_eff})
    logger.info("[family:generic] DONE")


def run_family_behavior(args, behavior: str) -> None:
    """One behavior family: stream slice -> fits + batteries + parity -> reap."""
    staging = Path(args.staging_root)
    done = family_done_path(staging, behavior)
    if done.is_file() and not args.force:
        logger.info("[family:%s] done sentinel present — skip", behavior)
        return
    if not family_done_path(staging, "generic").is_file():
        raise RuntimeError("family generic must complete first (retained core is its output)")
    assert_family_headroom(staging, behavior)
    revision = args.dataset_revision
    contexts_dir = stage_contexts_tree(staging, revision)
    ret = retained_dir(staging)
    fits_dir = ret / "fits"
    fits_dir.mkdir(exist_ok=True)
    fits = _fits2091()
    hf_token = os.environ.get("HF_TOKEN") or ""

    # ── stage: labeling tar slice (stream), greedy stores, probe store ──
    slice_dir = staging / f"labeling_{behavior}"
    slice_done = slice_dir / "_slice_done.json"
    if not slice_done.is_file():
        sl = _slice_mod()
        digest = sl.stream_slice(
            behavior,
            slice_dir,
            revision=revision,
            kinds=("t1", "context_end"),
            layers=LAYERS,
            token=hf_token,
            workers=args.stream_workers,
        )
        write_json_atomic(slice_done, digest)
    keep_store = store_basename_keep(("t1", "context_end"), LAYERS)
    greedy_dirs = {}
    for job in FAMILY_JOBS[behavior]:
        d = staging / f"greedy_{job}"
        stage_prefix_files(
            f"{HF_PREFIX}/capture_store/greedy_{job}", d, revision=revision, keep=keep_store
        )
        greedy_dirs[job] = d
    probe_dir = staging / f"parity_probe_{behavior}"
    stage_prefix_files(
        f"{HF_PREFIX}/capture_store/parity_probe_{behavior}",
        probe_dir,
        revision=revision,
        keep=keep_store,
    )

    banked = StoreView(slice_dir)
    greedy_views = {job: StoreView(d) for job, d in greedy_dirs.items()}
    probe_view = StoreView(probe_dir)

    # ── capture parity (MF-2: full coverage + deferred probe) ──
    job_rows = {job: load_job_contexts(contexts_dir, job) for job in FAMILY_JOBS[behavior]}
    parity = {
        "full_coverage": capture_parity_full(greedy_views, banked),
        "probe": capture_parity_probe(probe_view, banked),
    }
    write_json_atomic(ret / f"parity_{behavior}.json", parity)

    # ── DV columns ──
    banked_rows = load_banked_dv(behavior)
    banked_by_ctx = {str(r["context_id"]): r for r in banked_rows}
    try:
        greedy_payload = load_greedy_dv(Path(args.out_root), behavior)
        greedy_dv = greedy_dv_by_ctx(greedy_payload)
    except FileNotFoundError:
        greedy_dv = {}
        logger.warning("[family:%s] greedy DV absent — greedy DV columns skipped", behavior)

    picks: dict[str, dict] = {}
    s4_map: dict[str, str] = {}
    if behavior == "hallucination":
        all_family_cids = {
            str(r["context_id"]) for job in FAMILY_JOBS[behavior] for r in job_rows[job]
        }
        pick_k: dict[str, int] = {}
        for cid in all_family_cids:
            row = banked_by_ctx.get(cid) or {}
            if row.get("per_rollout_scores"):
                k = fits.s1_single_draw_pick(cid, row["per_rollout_scores"]).k
            else:
                k = int(rng_for(f"s4::{cid}").integers(K_ROLLOUTS))
            picks[cid] = {"k": k, "dv": None, "dv_included": True}
            pick_k[cid] = k
        packed = stage_packed_lookup(behavior, staging, revision, all_family_cids)
        abstain = fits.load_banked_abstain_scores(stage_per_draw_tables(staging, revision))
        s4_map = s4_labels(pick_k, packed, abstain)
        for cid, lab in s4_map.items():
            picks[cid]["dv"] = (
                None
                if lab in ("unjudged", "missing_packed_row")
                else (100.0 if lab == "fabricated" else 0.0)
            )
    else:
        for r in banked_rows:
            if r.get("per_rollout_scores"):
                p = fits.s1_single_draw_pick(str(r["context_id"]), r["per_rollout_scores"])
                picks[p.context_id] = {"k": p.k, "dv": p.score, "dv_included": p.dv_included}

    # ── retained generic core (family A output) ──
    core = np.load(ret / "generic_core.npz", allow_pickle=False)
    control = {
        "x_pool": core["x"],
        "y_pool": {r: core[f"y_{r}"] for r in REGIMES},
        "selected_lambda": json.loads((ret / "generic_control_lambdas.json").read_text()),
    }
    core_ids = core["ids"].tolist()
    core_groups = core["groups"].astype(object)
    maps_path = ret / "maps" / "context_end__ufull.npz"
    rb_arr, rb_names = _store_io().load_rb_bank()
    rb_by_layer = {ly: rb_arr[ly, rb_names.index(behavior), :] for ly in LAYERS}

    # ── per-setting loop (train rung + OOD rungs) ──
    blocks: dict[str, dict] = {}
    grids: dict[str, dict] = {}
    battery = BatteryTimer(args.boot_b)
    b_eff: int | None = None
    for job in FAMILY_JOBS[behavior]:
        rows = job_rows[job]
        gk = {str(r["context_id"]): str(r["group_key"]) for r in rows}
        qk = {str(r["context_id"]): str((r.get("meta") or {}).get("question_key")) for r in rows}
        is_evil_train = job == "evil_train"
        gv = greedy_views[job]
        present = set(banked.rows_by_ctx) & set(gv.rows_by_ctx)
        pool_c = [str(r["context_id"]) for r in rows if r["split"] == "pool"]
        eval_c = [str(r["context_id"]) for r in rows if r["split"] == "eval"]
        n_planned = len(pool_c) + len(eval_c)
        pool_c = [c for c in pool_c if c in present]
        eval_c = [c for c in eval_c if c in present]
        if args.limit:
            pool_c, eval_c = pool_c[: args.limit], eval_c[: args.limit]
        all_c = pool_c + eval_c
        if len(all_c) < n_planned:
            logger.warning(
                "[family:%s] %s: %d/%d planned contexts resolve in both stores",
                behavior,
                job,
                len(all_c),
                n_planned,
            )

        sm_rows = [
            banked_by_ctx[c] for c in all_c if banked_by_ctx.get(c, {}).get("per_rollout_scores")
        ]
        scores, score_ids = per_rollout_score_matrix(sm_rows) if sm_rows else (None, None)

        def _run_block(bb: int):
            return setting_stat_block(
                setting=job,
                behavior=behavior,
                cids=all_c,
                banked=banked,
                greedy=gv,
                scores=scores,
                score_ids=score_ids,
                groups=[gk[c] for c in all_c],
                groups2=[qk[c] for c in all_c] if is_evil_train else None,
                b_boot=bb,
                rb_vec_by_layer=rb_by_layer,
            )

        if b_eff is None:
            b_eff = battery.resolve(_run_block)
        blocks[job] = _run_block(b_eff)
        if behavior == "hallucination" and "r5" not in blocks[job]:
            r5_def = hallu_own_rung_r5(
                [banked_by_ctx[c] for c in all_c if c in banked_by_ctx], all_c
            )
            if r5_def:
                blocks[job]["r5"] = r5_def

        # ── R4 fits for this setting ──
        admix = fits.assemble_pool_ids(
            [i for i in core_ids if not i.startswith("lmsys::")],
            [i for i in core_ids if i.startswith("lmsys::")],
            pool_c,
            pool_tag=job,
            n_wc=min(fits.GENERIC_WC, sum(1 for i in core_ids if not i.startswith("lmsys::"))),
            n_lmsys=min(fits.GENERIC_LMSYS, sum(1 for i in core_ids if i.startswith("lmsys::"))),
            u_pool=len(core_ids),
        )
        fits.assert_group_disjoint([gk[c] for c in pool_c], [gk[c] for c in eval_c], where=job)
        id_to_row = {cid: i for i, cid in enumerate(core_ids)}
        keep_core = [id_to_row[c] for c in admix.pool_ids if c in id_to_row]
        admix_ctx = [c for c in admix.pool_ids if c not in id_to_row]
        pool_x = np.concatenate(
            [
                core["x"][:, np.array(keep_core, dtype=int), :],
                np.stack([banked.vc(admix_ctx, ly) for ly in LAYERS]),
            ],
            axis=1,
        )
        pool_y = {}
        for regime in REGIMES:
            adm = np.stack(
                [build_regime_targets(admix_ctx, banked, gv, ly, picks)[regime] for ly in LAYERS]
            )
            pool_y[regime] = np.concatenate(
                [core[f"y_{regime}"][:, np.array(keep_core, dtype=int), :], adm], axis=1
            )
        pool_groups = np.concatenate(
            [
                core_groups[np.array(keep_core, dtype=int)],
                np.array([gk[c] for c in admix_ctx], dtype=object),
            ]
        )
        eval_x = np.stack([banked.vc(eval_c, ly) for ly in LAYERS])
        eval_y = {
            regime: np.stack(
                [build_regime_targets(eval_c, banked, gv, ly, picks)[regime] for ly in LAYERS]
            )
            for regime in REGIMES
        }
        grid = fit_setting_grids(
            setting=job,
            pool_spec={
                k: v for k, v in admix.to_json().items() if k not in ("pool_ids", "control_ids")
            },
            pool_x=pool_x,
            pool_y=pool_y,
            eval_x=eval_x,
            eval_y=eval_y,
            groups=pool_groups,
            eval_groups=[gk[c] for c in eval_c],
            fits_dir=fits_dir,
            control=control,
        )
        grid["map963k"] = map963k_reference(maps_path, eval_x, eval_y)
        preds = grid.pop("_predictions")

        # ── behavioral rho columns (R4-b) ──
        avg_dv = {
            c: banked_by_ctx[c].get("dv")
            for c in all_c
            if banked_by_ctx.get(c, {}).get("dv") is not None
        }
        dv_cols = {
            "avg_k5": np.array([avg_dv.get(c, np.nan) for c in eval_c], dtype=np.float64),
            "single": np.array(
                [
                    picks[c]["dv"] if c in picks and picks[c].get("dv") is not None else np.nan
                    for c in eval_c
                ],
                dtype=np.float64,
            ),
        }
        if greedy_dv:
            dv_cols["greedy"] = np.array([greedy_dv.get(c, np.nan) for c in eval_c])
        li = LAYERS.index(HEADLINE_LAYER)
        rho_boot = GroupBootstrap([gk[c] for c in eval_c], b_eff, f"{job}::rho")
        labeled_pool = [c for c in pool_c if c in avg_dv]
        half_a = half_b = None
        if scores is not None and score_ids:
            sidx = {c: i for i, c in enumerate(score_ids)}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                half_a = np.array(
                    [
                        np.nanmean(scores[sidx[c], [0, 2, 4]]) if c in sidx else np.nan
                        for c in eval_c
                    ]
                )
                half_b = np.array(
                    [np.nanmean(scores[sidx[c], [1, 3]]) if c in sidx else np.nan for c in eval_c]
                )
        ro = behavioral_readouts(
            setting=job,
            layer_idx=li,
            layer=HEADLINE_LAYER,
            rb_vec=rb_by_layer[HEADLINE_LAYER],
            eval_x=eval_x,
            eval_v5_mean=banked.t1_stack(eval_c, HEADLINE_LAYER).mean(axis=1),
            predictions=preds,
            dv_cols=dv_cols,
            labeled_pool_x=(
                np.stack([banked.vc(labeled_pool, ly) for ly in LAYERS]) if labeled_pool else None
            ),
            labeled_pool_va=(
                np.stack([banked.t1_stack(labeled_pool, ly).mean(axis=1) for ly in LAYERS])
                if labeled_pool
                else None
            ),
            labeled_pool_dv=(
                {"avg_k5": np.array([avg_dv[c] for c in labeled_pool])} if labeled_pool else None
            ),
            pool_groups=(
                np.array([gk[c] for c in labeled_pool], dtype=object) if labeled_pool else None
            ),
            dv_half_a=half_a,
            dv_half_b=half_b,
            boot=rho_boot,
        )
        pcs = ro.pop("_percontext_scores")
        grid["behavioral_rho_L19"] = ro

        # ── R3 commonality over the eval set ──
        v5e = banked.t1_stack(eval_c, HEADLINE_LAYER)
        sig_tot_e = np.sqrt(
            np.mean(
                np.linalg.norm(
                    v5e.astype(np.float64) - v5e.astype(np.float64).mean(axis=1, keepdims=True),
                    axis=2,
                )
                ** 2,
                axis=1,
            )
        )
        projk_e = np.einsum("nkd,d->nk", v5e.astype(np.float64), rb_by_layer[HEADLINE_LAYER])
        sd_e = np.full(len(eval_c), np.nan)
        mu_e = np.array([avg_dv.get(c, np.nan) for c in eval_c])
        if scores is not None and score_ids:
            sidx = {c: i for i, c in enumerate(score_ids)}
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                sd_e = np.array(
                    [
                        np.sqrt(np.nanvar(scores[sidx[c]], ddof=0)) if c in sidx else np.nan
                        for c in eval_c
                    ]
                )
        with np.errstate(invalid="ignore", divide="ignore"):
            ceil_e = np.sqrt(np.clip(mu_e * (100 - mu_e), 0, None))
            p_e = np.where(ceil_e > 0, sd_e / ceil_e, np.nan)
        blocks[job]["r3_commonality"] = commonality_block(
            percontext_scores=pcs,
            dv_avg=dv_cols["avg_k5"],
            sigma_defs={
                "sigma_a_total": sig_tot_e,
                "sigma_a_proj": projk_e.std(axis=1, ddof=0),
            },
            p_arr=p_e,
            boot=rho_boot,
        )

        # ── R4-c: low/high sigma_A median split (within-column contrasts, A4) ──
        med = float(np.median(sig_tot_e))
        split_grids = {}
        for name, mask in (("low_sigma", sig_tot_e <= med), ("high_sigma", sig_tot_e > med)):
            split_grids[name] = {
                f"L{layer}": {
                    fr_: {
                        er: r2_score_rows(preds[fr_][lj][mask], eval_y[er][lj][mask])
                        for er in REGIMES
                    }
                    for fr_ in REGIMES
                }
                for lj, layer in enumerate(LAYERS)
            }
        grid["median_split"] = {
            "note": (
                "within-fixed-eval-column contrasts; the high-sigma half co-varies with "
                "length/truncation — no cross-column causal reading (plan A4)"
            ),
            "grids": split_grids,
            "n_low": int((sig_tot_e <= med).sum()),
            "n_high": int((sig_tot_e > med).sum()),
        }
        grids[job] = grid
        _dump_partial(
            staging,
            behavior,
            {"blocks": blocks, "grids": grids, "parity": parity, "battery_b_effective": b_eff},
        )
        logger.info(
            "[family:%s] setting %s done (pool=%d eval=%d)", behavior, job, len(pool_c), len(eval_c)
        )

    partial = {
        "blocks": blocks,
        "grids": grids,
        "parity": parity,
        "battery_b_effective": b_eff,
        "s4_label_counts": (
            {
                lab: sum(1 for v in s4_map.values() if v == lab)
                for lab in sorted(set(s4_map.values()))
            }
            if s4_map
            else None
        ),
    }
    _dump_partial(staging, behavior, partial)

    # ── reap family inputs (MF-1) ──
    to_reap = [slice_dir, probe_dir, *greedy_dirs.values()]
    if behavior == "hallucination":
        to_reap += [staging / "packed_hallucination", staging / "per_draw_hallucination"]
    for p in to_reap:
        reap(p)
    write_json_atomic(done, {"ts": time.time(), "b_effective": b_eff})
    logger.info("[family:%s] DONE", behavior)


# ── assembly ──────────────────────────────────────────────────────────────────
def phase_assemble(args) -> None:
    """Merge per-family partials into the §6.5 primary-deliverable JSONs."""
    staging = Path(args.staging_root)
    out_root = Path(args.out_root)
    ret = retained_dir(staging)
    partials = {}
    for family in FAMILIES:
        p = ret / f"partial_{family}.json"
        if not p.is_file():
            raise FileNotFoundError(
                f"partial missing for family {family}: {p} — run its family phase first"
            )
        partials[family] = json.loads(p.read_text())
    meta = {
        **_provenance_meta(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "layers": list(LAYERS),
        "headline_layer": HEADLINE_LAYER,
        "boot_b_requested": args.boot_b,
        "boot_b_effective": {f: partials[f].get("battery_b_effective") for f in FAMILIES},
    }

    # r1_dispersion.json
    r1 = {"meta": meta, "settings": {}}
    r1["settings"]["wildchat"] = partials["generic"]["blocks"]["wildchat"].get("r1")
    r1["settings"]["lmsys"] = {
        layer: {
            "summary": blk["dispersion_summary"],
            "boot_ci_median": blk["dispersion_boot_ci_median"],
            "median_answer_len": blk["median_answer_len"],
            "percontext": blk["percontext"],
        }
        for layer, blk in partials["generic"]["blocks"]["lmsys"]["per_layer"].items()
    }
    for b in BEHAVIORS:
        for job, blk in partials[b]["blocks"].items():
            r1["settings"][job] = blk.get("r1")
    r1["headline_note"] = (
        "generic-vs-trait headline reads off the matched-instrument WildChat bar; "
        "LMSYS bars shown for context only (plan A1); median answer length available "
        "for LMSYS (span_lens) — behavior-store medians deferred to unit D (raw text shards)"
    )
    write_json_atomic(out_root / "r1_dispersion.json", r1)

    # r2_delta.json (+ generic-vs-trait Holm contrasts on median Delta)
    r2 = {"meta": meta, "settings": {}}
    r2["settings"]["wildchat"] = partials["generic"]["blocks"]["wildchat"].get("r2")
    r2["settings"]["lmsys"] = {
        layer: {
            k: v
            for k, v in blk.items()
            if k.startswith(("median_delta", "delta_boot", "severe", "percontext", "deviations"))
        }
        for layer, blk in partials["generic"]["blocks"]["lmsys"]["per_layer"].items()
    }
    for b in BEHAVIORS:
        for job, blk in partials[b]["blocks"].items():
            r2["settings"][job] = blk.get("r2")
    diffs = {}
    wc_pc = (r2["settings"]["wildchat"] or {}).get(f"L{HEADLINE_LAYER}", {}).get("percontext")
    for b, job in (
        ("sycophancy", "syc_train"),
        ("hallucination", "hal_train"),
        ("evil", "evil_train"),
    ):
        tr = (r2["settings"].get(job) or {}).get(f"L{HEADLINE_LAYER}", {}).get("percontext")
        if not tr or not wc_pc:
            continue
        boot_t = GroupBootstrap([str(x) for x in tr["context_id"]], B_BOOT_DESCOPED, f"holm::{job}")
        boot_w = GroupBootstrap([str(x) for x in wc_pc["context_id"]], B_BOOT_DESCOPED, "holm::wc")
        diffs[f"{b}_train_minus_wildchat"] = boot_t.median(
            np.asarray(tr["delta"], dtype=np.float64)
        ) - boot_w.median(np.asarray(wc_pc["delta"], dtype=np.float64))
    r2["generic_vs_trait_contrasts_L19"] = holm_adjusted_cis(diffs) if diffs else None
    r2["mf2_weigh_note"] = (
        "any negative trait-side Delta is read against capture_parity.json + the "
        "WildChat(cross-campaign)-vs-LMSYS(same-campaign) triangulation BEFORE any #1073 "
        "correction posts (plan §6 MF-2 weigh rule — report, never a gate)"
    )
    write_json_atomic(out_root / "r2_delta.json", r2)

    # r3_moderators_<behavior>.json
    for b in BEHAVIORS:
        blocks = dict(partials[b]["blocks"])
        wc_blk = partials["generic"].get("wc_score_blocks", {}).get(b)
        if wc_blk:
            blocks["wildchat"] = wc_blk
        r3: dict = {
            "meta": meta,
            "behavior": b,
            "judge_draw_var": judge_draw_variance(b),
            "settings": {},
        }
        r3["judge_var_note"] = (
            "per-draw judge variance from REALIZED banked draw-matrix rows only "
            "(evil coverage 70% — plan A4)"
        )
        for setting, blk in blocks.items():
            if not blk.get("r3_inputs"):
                continue
            r3["settings"][setting] = {
                "ceilings": blk.get("r3_ceilings"),
                "guardrails": blk.get("r3_guardrails"),
                "commonality": blk.get("r3_commonality"),
                "moderators_percontext": blk["r3_inputs"],
            }
        # rho(sigma_A median, P spread) across settings (R3-c)
        sig_meds, p_spreads = [], []
        for _setting, blk in blocks.items():
            r3in = blk.get("r3_inputs")
            if not r3in:
                continue
            sig = r3in["sigma_defs"].get("sigma_a_total")
            if sig is None:
                continue
            sig_meds.append(float(np.nanmedian(np.asarray(sig, dtype=np.float64))))
            p_spreads.append(float(np.nanstd(np.asarray(r3in["p"], dtype=np.float64))))
        r3["rho_sigma_vs_p_spread_across_settings"] = (
            spearman(np.array(sig_meds), np.array(p_spreads)) if len(sig_meds) >= 3 else None
        )
        write_json_atomic(out_root / f"r3_moderators_{b}.json", r3)

    # r4_grids.json
    r4 = {"meta": meta, "settings": {"generic": partials["generic"]["r4_generic"]}}
    for b in BEHAVIORS:
        for job, grid in (partials[b].get("grids") or {}).items():
            r4["settings"][job] = grid
    r4["expectation_band_note"] = (
        "generic panel vs #1073's pre-fill is an EXPECTATION BAND, not an identity — "
        "estimator + n differ (plan W2); layer sets differ from #1073's per-trait "
        "read-out layers (plan W3)"
    )
    write_json_atomic(out_root / "r4_grids.json", r4)

    # r5_polarization.json
    r5 = {"meta": meta, "panels": {}}
    for b in BEHAVIORS:
        wc_blk = partials["generic"].get("wc_score_blocks", {}).get(b)
        if wc_blk and wc_blk.get("r5"):
            r5["panels"][f"{b}::wildchat"] = wc_blk["r5"]
        for job, blk in partials[b]["blocks"].items():
            if blk.get("r5"):
                r5["panels"][f"{b}::{job}"] = blk["r5"]
    write_json_atomic(out_root / "r5_polarization.json", r5)

    # capture_parity.json
    cp = {"meta": meta, "behaviors": {b: partials[b].get("parity") for b in BEHAVIORS}}
    cp["note"] = (
        "MF-2: full-coverage context_end cosines per rung + the deferred P2 probe "
        "(t1 + context_end at probe rollouts); analyzer-weighed report, not a kill gate"
    )
    write_json_atomic(out_root / "capture_parity.json", cp)
    logger.info("[assemble] wrote §6.5 deliverables under %s", out_root)


# ── import-check (module-level, NOT inside main — the #1739 shadow gotcha) ───
def _import_check() -> int:
    """Execute every deferred import + bind key call shapes (gotchas #606/#1332)."""
    import inspect

    fits = _fits2091()
    _stage2091()
    _store_io()
    _judging()
    _slice_mod()
    _cap1073()
    _common1073()
    from scripts.issue1073_neardup_sensitivity import neardup_cluster_ids  # noqa: F401
    from scripts.issue1739_map963k_readout import load_i1739_map  # noqa: F401

    from explore_persona_space.orchestrate import provenance  # noqa: F401
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom  # noqa: F401

    for fn, kwargs in [
        (fits.fit_pool_regime, dict(x_pool=0, y_pool=0, x_eval=0, y_eval=0, groups=0, fit_id="x")),
        (fits.select_lambda, dict(x_pool=0, y_pool=0, groups=0, where="x")),
        (fits.fit_predict_at_lambda, dict(x_train=0, y_train=0, x_eval=0, lam=0.1)),
        (
            fits.assemble_pool_ids,
            dict(generic_wc_ids=[], generic_lmsys_ids=[], admix_ids=[], pool_tag="x"),
        ),
        (fits.s1_single_draw_pick, dict(context_id="c", per_rollout_scores={})),
        (fits.s4_single_draw_label, dict(packed_row={}, abstain_scores={}, context_id="c", k=0)),
        (fits.load_banked_abstain_scores, dict(shard_paths=[])),
        (fits.build_inner_caches, dict(x_pool=0, groups=0)),
    ]:
        inspect.signature(fn).bind_partial(**kwargs)
    print("[import-check] OK: deferred imports resolved + call shapes bound", flush=True)
    return 0


# ── CLI ───────────────────────────────────────────────────────────────────────
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        default="all",
        choices=[
            "probe",
            "upload-judge-artifacts",
            "family",
            "assemble",
            "all",
            "banked-smoke",
            "import-check",
        ],
    )
    ap.add_argument("--family", choices=list(FAMILIES), help="with --phase family")
    ap.add_argument("--behavior", help="banked-smoke behavior (default sycophancy)")
    ap.add_argument("--staging-root", type=Path, default=DEFAULT_STAGING_ROOT)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--dataset-revision", default="main")
    ap.add_argument("--boot-b", type=int, default=B_BOOT_DEFAULT)
    ap.add_argument("--stream-workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0, help="smoke: cap contexts per setting")
    ap.add_argument("--force", action="store_true", help="ignore family done sentinels")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--skip-disk-probe", action="store_true")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.phase == "import-check":
        return _import_check()
    if args.phase == "banked-smoke":
        return phase_banked_smoke(args)
    if args.phase == "upload-judge-artifacts":
        phase_upload_judge_artifacts(args)
        return 0
    if not args.skip_disk_probe:
        probe_disk(Path(args.staging_root))
    if args.phase == "probe":
        return 0
    if args.phase in ("family", "all"):
        fams = [args.family] if (args.phase == "family" and args.family) else list(FAMILIES)
        for fam in fams:
            if fam == "generic":
                run_family_generic(args)
            else:
                run_family_behavior(args, fam)
    if args.phase in ("assemble", "all"):
        phase_assemble(args)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit before C-extension finalize (gotchas.md PyGILState)
