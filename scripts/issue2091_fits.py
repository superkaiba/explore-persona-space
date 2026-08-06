"""P4 fits driver for issue #2091 — dof-capped selection WRAPPER around the #1739 ridge cores.

The #1739 batched cores (``experiments/issue_1739/fits.ridge_layer_batched_auto`` +
siblings) are the layout-compatible batched kernels but implement PLAIN per-slice
GCV — no dof cap, no selector switch, no refusal guard, no ``groups`` kwarg (plan
§12 A9; verified grep) — so they are NEVER called bare for lambda SELECTION. This
module is the thin selection layer the plan pins (§4.2 P4-fits / §11 ridge
decision), carrying the #825 discipline:

  * explicit lambda grid (``LAMBDA_GRID`` — the #825 ``LAMBDAS`` logspace(-2, 4, 13));
  * inner-group-CV lambda selection on ``group_key`` within each pool
    (``N_INNER_FOLDS = 5`` inner folds; the #1335 r8 selector, REUSED byte-for-byte
    via ``issue825_fit_cells._prep_inner_lambda`` + ``_inner_cv_rss_curve``);
  * dof cap 0.9 — a candidate lambda whose effective dof exceeds
    ``DOF_CAP x n_inner_train`` on ANY usable inner fold is EXCLUDED from selection
    (mirroring ``issue825_fit_cells.GCV_DOF_CAP``);
  * a pure-GCV-at-n_train<d refusal mirroring ``issue825_fit_cells._refuse_unguarded_gcv``
    (invoked verbatim on the GCV fallback path);
  * final fit on the full pool at the selected lambda: ONE 1-element-grid core call
    per (pool x regime x layer) — 27 wrapper calls x 3 layers = 81 solves;
  * per-fit selector + selected-lambda diagnostics persisted (SELECTOR_LOG-style);
  * the mapping-baselines pair per fit (identity+learned-bias + kNN retrieval,
    ``experiments/issue_1739/fits.map_diagnostics`` — euclidean + cosine, k in {1, 5},
    chance = k/n_pool carried by the helper).

Standardization-convention note (load-bearing for the parity test): the #825 fit
path standardizes X with the SAMPLE std (``torch.std`` default, correction=1) while
the #1739 cores use the POPULATION std ("twin parity" with the #779 dual parent).
The two fits are related EXACTLY by a lambda rescale: with c = (ntr-1)/ntr,
``pred_825(lam) == pred_core(lam / c)`` (dual algebra: Gram_825 = c * Gram_core).
Lambda SELECTION under one convention therefore corresponds to a ~ntr/(ntr-1)
rescale under the other — negligible against the grid spacing (x ~3.16) — and the
parity unit test (``tests/test_issue2091_fits_parity.py``) pins both the selected-
lambda set (exact) and the held-out predictions (via the rescale identity) against
``issue825_fit_cells.heldout_r2_sweep`` on a small synthetic cell.

Pool construction (plan §4.2): U = 4,000 fixed; generic core = 1,500 WildChat-pool
+ 2,500 LMSYS-pool rows; admixed pools REPLACE generic rows LMSYS-first (constant
U); realized f_u recorded per cell, never silently degraded; the matched
all-generic control is the 4,000-row generic core itself. Pool/eval stay
group-disjoint by the staging-time designation (asserted here). S1/S4 single-draw
dispositions and the S5 pool-side readout filter live here too, so unit C's
analysis driver can consume one module.

This module exposes clean functions for the analysis driver (unit C): the
SEQUENTIAL per-family stage->consume->reap loop, the capture-parity read, and the
statistical batteries are deliberately NOT here.

CLI (unit-scoped smokes; the production P4 entry is driven by the analysis script):
  uv run python scripts/issue2091_fits.py --mode import-check
  uv run python scripts/issue2091_fits.py --mode pilot-synthetic [--out DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) bind BEFORE numpy/torch import

import numpy as np  # noqa: E402

logger = logging.getLogger(__name__)


def _ensure_repo_root_on_syspath() -> Path:
    """Script-mode sys.path guard (gotchas.md #823): repo root for scripts.* imports."""
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue825_fit_cells.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

TASK_ID = 2091

# ── pins (plan §10/§11) ───────────────────────────────────────────────────────
SEED = 20910  # plan §10: staging + subsample + single-draw share one seed
# Explicit lambda grid == issue825_fit_cells.LAMBDAS (np.logspace(-2, 4, 13)),
# asserted equal at first use (never a silently drifted copy).
LAMBDA_GRID: tuple[float, ...] = tuple(float(x) for x in np.logspace(-2, 4, 13))
N_INNER_FOLDS = 5  # plan §4.2: inner-group-CV on group_key, 5 inner folds
DOF_CAP: float = 0.9  # mirrors issue825_fit_cells.GCV_DOF_CAP (#1887 default)
LAYERS: tuple[int, ...] = (14, 19, 26)  # frozen read-out set; headline L19 (§11)
HEADLINE_LAYER = 19
REGIMES: tuple[str, ...] = ("greedy", "avg_k5", "single")
K_ROLLOUTS = 5  # banked rollout grain (#1739)

# Per-rung pool rule (plan v5 §4.2/§4.3/§11): U_rung = max(U_FLOOR, 2 * A_rung),
# A_rung = min(U_rung // 2, len(admix_ids)). U_FLOOR is the well-posedness floor
# (d = 3,584, so U must exceed d — a smaller U is estimator-degenerate, #1701/#1887);
# the rule honors the registered f_u = 0.5 exactly wherever a rung can supply it
# (2 * A_rung >= U_FLOOR) and falls back to U = U_FLOOR where it cannot. On this
# round's realized admix pools (335..1003, all < 2,000) it evaluates to U = 4,000
# on every rung — identical to the previous fixed U_POOL = 4000 behavior.
U_FLOOR = 4000
GENERIC_WC = 1500
GENERIC_LMSYS = 2500
TARGET_F_U = 0.5  # registered target; realized f_u reported per cell
KNN_KS: tuple[int, ...] = (1, 5)

# SELECTOR_LOG-style compact telemetry (mirrors issue825_fit_cells.SELECTOR_LOG).
SELECTOR_LOG: dict[str, dict[str, int]] = {}

_UNSET = object()


def _log_selector(selector: str, lam: float) -> None:
    d = SELECTOR_LOG.setdefault(selector, {})
    k = f"{lam:.6g}"
    d[k] = d.get(k, 0) + 1


def _fit825():
    """The #825 discipline module (selection machinery reused byte-for-byte)."""
    _ensure_repo_root_on_syspath()
    import scripts.issue825_fit_cells as fit825

    return fit825


def _fits1739():
    """The #1739 batched ridge cores + map diagnostics."""
    from explore_persona_space.experiments.issue_1739 import fits as fits1739

    return fits1739


def _stable_hash64(s: str) -> int:
    return int.from_bytes(hashlib.sha256(s.encode("utf-8")).digest()[:8], "little")


# ── per-rollout score parsing (A3: zero-padded k00..k04 keys) ────────────────
_K_KEY_RE = re.compile(r"^k(\d{2})$")


def parse_per_rollout_scores(d: dict) -> dict[int, float | None]:
    """Parse a banked ``per_rollout_scores`` dict — keys are ZERO-PADDED ``kNN``.

    Plan §12 A3: every banked sycophancy/evil/WildChat row carries 5 keys
    ``k00..k04`` (None where the judge dropped every draw). A parser written to
    ``k0..k4`` KeyErrors; this one fail-louds on any non-``kNN`` key instead of
    silently skipping it.
    """
    out: dict[int, float | None] = {}
    for key, val in d.items():
        m = _K_KEY_RE.fullmatch(str(key))
        if m is None:
            raise ValueError(f"per_rollout_scores key {key!r} is not zero-padded 'kNN' (A3)")
        out[int(m.group(1))] = None if val is None else float(val)
    return out


# ── S1: single stochastic draw disposition ────────────────────────────────────
@dataclass(frozen=True)
class S1Pick:
    """Fixed-seed single-draw pick for one context (plan §4.2 S1).

    The SAME index ``k`` defines BOTH the single-draw vector target (that
    rollout's t1 row) and the single-draw judged DV — never decoupled.
    ``dv_included=False`` marks an all-None context: EXCLUDED from the
    single-draw DV column (reported per rung), vector-side draw over 0..K-1.
    """

    context_id: str
    k: int
    dv_included: bool
    score: float | None


def s1_single_draw_pick(
    context_id: str,
    per_rollout_scores: dict,
    *,
    k_rollouts: int = K_ROLLOUTS,
    seed: int = SEED,
) -> S1Pick:
    """Deterministic per-context draw among rollouts with a non-None judged score."""
    scores = (
        per_rollout_scores
        if per_rollout_scores and all(isinstance(k, int) for k in per_rollout_scores)
        else parse_per_rollout_scores(per_rollout_scores)
    )
    for k in scores:
        if not 0 <= int(k) < k_rollouts:
            raise ValueError(f"{context_id}: rollout index {k} outside 0..{k_rollouts - 1}")
    rng = np.random.default_rng([seed, _stable_hash64(context_id)])
    kept = sorted(int(k) for k, s in scores.items() if s is not None)
    if kept:
        k = int(kept[int(rng.integers(len(kept)))])
        return S1Pick(context_id, k, True, float(scores[k]))
    k = int(rng.integers(k_rollouts))
    return S1Pick(context_id, k, False, None)


# ── S4: hallucination single-draw three-way join (no new judge calls) ─────────
def load_banked_abstain_scores(
    shard_paths: Iterable[Path],
) -> dict[tuple[str, int], float | None]:
    """(context_id, rollout_k) -> mean banked abstain-judge score (or None).

    Reads the #1739 ``per_draw_<behavior>.shardNN.jsonl`` tables (schema:
    context_id, rollout_k, draw_idx, score, parse_ok, drop_class, transport,
    wave, judge_max_tokens). When a rollout carries rows from more than one
    wave, the MAX ``judge_max_tokens`` wave wins (the rejudge-800 wave
    superseded the base-400 draws in the banked labeling merge). A rollout
    whose winning wave kept no draws maps to None (drop-never-coerce).

    Text-mode line iteration, never ``.splitlines()`` (gotchas.md JSONL rule).
    """
    acc: dict[tuple[str, int], dict[int, list[float]]] = {}
    for path in shard_paths:
        with Path(path).open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                key = (str(row["context_id"]), int(row["rollout_k"]))
                budget = int(row.get("judge_max_tokens") or 0)
                waves = acc.setdefault(key, {})
                kept = waves.setdefault(budget, [])
                if row.get("score") is not None:
                    kept.append(float(row["score"]))
    out: dict[tuple[str, int], float | None] = {}
    for key, waves in acc.items():
        budget = max(waves)
        kept = waves[budget]
        out[key] = float(sum(kept) / len(kept)) if kept else None
    return out


def s4_single_draw_label(
    packed_row: dict,
    abstain_scores: dict[tuple[str, int], float | None],
    *,
    context_id: str,
    k: int,
) -> str:
    """Three-way label for the picked banked rollout k(c) — deterministic, no judge.

    Alias-match on the packed shard's completion + answer_aliases first
    (label ``correct``); otherwise the BANKED abstain-rubric score for that
    specific rollout, classified by ``judging.three_way_classify``. A rollout
    with no banked judged draws is ``unjudged`` (excluded + counted upstream).
    """
    from explore_persona_space.experiments.issue_1739 import judging

    aliases = packed_row.get("answer_aliases") or []
    if not aliases:
        raise ValueError(f"{context_id}: packed row has no answer_aliases (A8 contract)")
    if judging.alias_correct(packed_row["completion"], aliases):
        return "correct"
    return judging.three_way_classify(False, abstain_scores.get((context_id, int(k))))


# ── pool assembly (plan §4.2 pools) ───────────────────────────────────────────
@dataclass
class PoolSpec:
    """One admixed pool + its matched all-generic control (id-level)."""

    pool_tag: str
    pool_ids: list[str]
    control_ids: list[str]  # the full generic core (== U rows on the U_FLOOR branch)
    n_admix: int
    f_u_realized: float
    f_u_target: float
    n_removed_lmsys: int
    n_removed_wc: int
    u_pool: int  # realized per-rung U (max(u_floor, 2 * A_rung))
    u_pool_realized: int  # alias of u_pool for per-cell reporting (plan §4.2)
    u_floor: int  # the well-posedness floor the rung was derived under
    seed: int

    def to_json(self) -> dict:
        d = asdict(self)
        # id lists can be long; keep them (unit C persists the spec next to fits)
        return d


def _seeded_subset(ids: list[str], k: int, rng: np.random.Generator, *, what: str) -> list[str]:
    if len(ids) < k:
        raise ValueError(f"{what}: need {k} ids, have {len(ids)}")
    if len(ids) == k:
        return list(ids)
    idx = rng.choice(len(ids), size=k, replace=False)
    return [ids[i] for i in sorted(int(i) for i in idx)]


def assemble_pool_ids(
    generic_wc_ids: Sequence[str],
    generic_lmsys_ids: Sequence[str],
    admix_ids: Sequence[str],
    *,
    pool_tag: str,
    n_wc: int = GENERIC_WC,
    n_lmsys: int = GENERIC_LMSYS,
    u_floor: int = U_FLOOR,
    seed: int = SEED,
) -> PoolSpec:
    """Admix-by-REPLACEMENT pool with per-rung U (LMSYS removed first, then WildChat).

    Plan v5 §4.2/§4.3: ``U_rung = max(u_floor, 2 * A_rung)`` with
    ``A_rung = min(U_rung // 2, len(admix_ids))`` — the rule honors the
    registered f_u = 0.5 EXACTLY wherever the rung's admix can supply it
    (``2 * A >= u_floor``) and falls back to the well-posedness floor
    ``U = u_floor`` where it cannot (U must exceed d = 3,584; #1701/#1887).
    Generic core = ``n_wc`` WildChat-pool + ``n_lmsys`` LMSYS-pool rows; the
    pool draws ``U - A`` generic rows from it (equivalently: removes
    ``(n_wc + n_lmsys) - (U - A)`` core rows LMSYS-first, then inserts the A
    admix rows), so the generic core is a FINITE supply: ``U - A`` exceeding
    ``n_wc + n_lmsys`` fails loud (never silently truncated). Realized
    f_u = A/U is RECORDED, never silently degraded. The control is the full
    generic core itself (== U rows on the floor branch; on the 2A branch the
    finite core cannot supply a U-row all-generic control, so the control
    stays the ``n_wc + n_lmsys``-row core). ``admix_ids`` are UNLABELED map
    rows only — group-disjointness from every eval context is the staging
    split's job; assert via ``assert_group_disjoint``.

    On the floor branch this is draw-for-draw identical to the previous fixed
    ``U_POOL = 4000`` implementation (same RNG consumption order: core-wc,
    core-lmsys, admix, remove-lmsys, remove-wc), pinned by
    ``tests/test_issue2091_pool_rule.py`` golden digests.
    """
    wc = sorted(dict.fromkeys(str(i) for i in generic_wc_ids))
    lm = sorted(dict.fromkeys(str(i) for i in generic_lmsys_ids))
    ad = sorted(dict.fromkeys(str(i) for i in admix_ids))
    overlap = (set(wc) & set(lm)) | (set(wc) & set(ad)) | (set(lm) & set(ad))
    if overlap:
        raise ValueError(f"pool id families overlap (n={len(overlap)}): {sorted(overlap)[:3]}")

    n_core = n_wc + n_lmsys
    u_pool = max(u_floor, 2 * len(ad))
    a = min(u_pool // 2, len(ad))  # == len(ad) by construction; keep the registered formula
    n_generic_needed = u_pool - a
    if n_generic_needed > n_core:
        raise ValueError(
            f"{pool_tag}: generic supply over-draw — U={u_pool} needs {n_generic_needed} "
            f"generic rows but the core has only {n_wc}+{n_lmsys}={n_core} "
            "(fail loud, never silently truncate; plan §4.2 pools)"
        )
    n_removed = n_core - n_generic_needed

    rng = np.random.default_rng([seed, _stable_hash64(f"pool::{pool_tag}")])
    core_wc = _seeded_subset(wc, n_wc, rng, what=f"{pool_tag}/generic-wc")
    core_lm = _seeded_subset(lm, n_lmsys, rng, what=f"{pool_tag}/generic-lmsys")
    control_ids = core_lm + core_wc

    admix_sel = _seeded_subset(ad, a, rng, what=f"{pool_tag}/admix") if a else []
    r_lm = min(n_removed, n_lmsys)
    r_wc = n_removed - r_lm
    removed_lm = set(_seeded_subset(core_lm, r_lm, rng, what=f"{pool_tag}/remove-lmsys"))
    removed_wc = set(_seeded_subset(core_wc, r_wc, rng, what=f"{pool_tag}/remove-wc"))
    pool_ids = (
        [i for i in core_lm if i not in removed_lm]
        + [i for i in core_wc if i not in removed_wc]
        + admix_sel
    )
    if len(pool_ids) != u_pool:
        raise RuntimeError(f"{pool_tag}: assembled {len(pool_ids)} != U={u_pool}")
    return PoolSpec(
        pool_tag=pool_tag,
        pool_ids=pool_ids,
        control_ids=control_ids,
        n_admix=a,
        f_u_realized=a / u_pool,
        f_u_target=TARGET_F_U,
        n_removed_lmsys=r_lm,
        n_removed_wc=r_wc,
        u_pool=u_pool,
        u_pool_realized=u_pool,
        u_floor=u_floor,
        seed=seed,
    )


def assert_group_disjoint(
    pool_groups: Iterable[str], eval_groups: Iterable[str], *, where: str
) -> None:
    """Fail loud when pool-side and eval-side groups overlap (staging invariant)."""
    inter = {str(g) for g in pool_groups} & {str(g) for g in eval_groups}
    if inter:
        raise RuntimeError(
            f"{where}: pool/eval group overlap (n={len(inter)}, e.g. {sorted(inter)[:3]}) — "
            "the staging-time split must be group-disjoint"
        )


def s5_pool_side_filter(labeled_rows: list[dict], pool_side_groups: Iterable[str]) -> list[dict]:
    """S5: restrict the supervised-readout labeled set L to POOL-side groups.

    Every supervised readout (context-side AND answer-side) trains ONLY on
    labeled rows whose ``group_key`` is pool-side under the staging-time
    designation; eval rows are eval-side groups only (plan §4.2 S5).
    """
    keep = {str(g) for g in pool_side_groups}
    return [r for r in labeled_rows if str(r.get("group_key")) in keep]


# ── lambda selection (the #825 discipline) ────────────────────────────────────
@dataclass
class LambdaSelection:
    """Per-fit selector diagnostics (SELECTOR_LOG-style, persisted per fit)."""

    best_lambda: float
    selector: str  # "inner-group-cv" | "gcv-fallback"
    grid: list[float]
    rss_curve: list[float] | None  # inner-CV summed validation RSS per grid lambda
    gcv_curve: list[float] | None  # fallback-path GCV values per grid lambda
    excluded_lambdas: list[float]  # dof-cap exclusions (selection never picks these)
    dof_cap: float | None
    n_inner_folds_used: int
    inner_fold_ntr: list[int]
    n_train: int
    d: int
    where: str = ""

    def to_json(self) -> dict:
        return asdict(self)


def build_inner_caches(
    x_pool: np.ndarray,
    groups: np.ndarray,
    *,
    n_inner: int = N_INNER_FOLDS,
    seed: int = SEED,
    device: str | None = None,
) -> list[dict] | None:
    """Y-independent inner-group-CV caches for ONE (pool, layer) design matrix.

    Thin wrapper over ``issue825_fit_cells._prep_inner_lambda`` (the #1335 r8
    machinery, reused byte-for-byte) so the caches can be SHARED across the
    three regime targets of one pool x layer (the caches depend only on X).
    ``device=None`` keeps the #825 ``_fit_device()`` routing.
    """
    import torch

    fit825 = _fit825()
    dev = None if device is None else torch.device(device)
    return fit825._prep_inner_lambda(np.asarray(x_pool), np.asarray(groups), n_inner, seed, dev)


def _dof_cap_exclusions(
    inner_caches: list[dict], grid: np.ndarray, dof_cap: float | None
) -> tuple[np.ndarray, list[int]]:
    """Per-grid-lambda exclusion mask: dof(lam) > dof_cap * n_inner_train on ANY fold."""
    fold_ntr = [int(ic["w"].shape[0]) for ic in inner_caches]
    excluded = np.zeros(len(grid), dtype=bool)
    if dof_cap is None:
        return excluded, fold_ntr
    for ic in inner_caches:
        w = ic["w"]
        n_fi = int(w.shape[0])
        for i, lam in enumerate(grid):
            dof = float((w / (w + float(lam))).sum())
            if dof > dof_cap * n_fi:
                excluded[i] = True
    return excluded, fold_ntr


def _gcv_select_capped(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    *,
    grid: np.ndarray,
    dof_cap: float | None,
    where: str,
) -> LambdaSelection:
    """Dof-capped GCV fallback (mirrors the #825 serial scan) with the n<d refusal.

    Runs ONLY when the inner-group caches are unbuildable (< 2 usable inner
    group folds — a should-never-happen regime for the #2091 pools). The
    pure-GCV-at-n_train<d refusal is ``issue825_fit_cells._refuse_unguarded_gcv``
    invoked verbatim with THIS wrapper's cap.
    """
    fit825 = _fit825()
    x = np.asarray(x_pool)
    y = np.asarray(y_pool)
    ntr, d = x.shape
    fit825._refuse_unguarded_gcv(
        ntr=ntr, d=d, cap=dof_cap, legacy_ok=False, where=f"issue2091_fits.{where}"
    )
    cache = fit825._prep_fold(x, x[:1])  # eval slice unused; w/V/ntr are what we need
    w = cache["w"]
    ytr = fit825._as_f64_on(y, w.device)
    ymu = ytr.mean(0)
    ytr_c = ytr - ymu
    vty = cache["V"].T @ ytr_c
    sq_vty = (vty**2).sum(1)
    tot = float((ytr_c**2).sum())
    gcv_curve: list[float] = []
    excluded: list[float] = []
    best_lam: float | None = None
    best_gcv = float("inf")
    for lam in grid:
        lam_f = float(lam)
        filt = w / (w + lam_f)
        dof = float(filt.sum())
        if dof_cap is not None and dof > dof_cap * ntr:
            excluded.append(lam_f)
            gcv_curve.append(float("inf"))
            continue
        rss = tot - float(((2 * filt - filt**2) * sq_vty).sum())
        denom = (ntr - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        gcv_curve.append(float(gcv))
        if gcv < best_gcv:
            best_gcv = gcv
            best_lam = lam_f
    if best_lam is None:
        raise RuntimeError(
            f"{where}: every grid lambda excluded/degenerate under dof_cap={dof_cap} "
            f"(ntr={ntr}, d={d}) — no admissible lambda; widen the grid or revisit the pool"
        )
    return LambdaSelection(
        best_lambda=best_lam,
        selector="gcv-fallback",
        grid=[float(v) for v in grid],
        rss_curve=None,
        gcv_curve=[float(v) for v in gcv_curve],
        excluded_lambdas=excluded,
        dof_cap=dof_cap,
        n_inner_folds_used=0,
        inner_fold_ntr=[],
        n_train=ntr,
        d=d,
        where=where,
    )


def select_lambda(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    groups: np.ndarray | None,
    *,
    lambdas: Sequence[float] = LAMBDA_GRID,
    n_inner: int = N_INNER_FOLDS,
    seed: int = SEED,
    dof_cap: float | None = DOF_CAP,
    device: str | None = None,
    inner_caches: object = _UNSET,
    where: str = "",
) -> LambdaSelection:
    """Inner-group-CV lambda selection with the dof cap + n<d refusal (plan §4.2).

    ``inner_caches`` may be passed to share the Y-independent eigh work across
    the three regime targets of one pool x layer; ``_UNSET`` builds them from
    (x_pool, groups, n_inner, seed) — byte-identical to the #825 machinery.
    """
    import torch

    fit825 = _fit825()
    grid = np.asarray(lambdas, dtype=np.float64)
    fit825._validate_lambda_grid(grid)
    x = np.asarray(x_pool)
    ntr, d = x.shape
    caches = inner_caches
    if caches is _UNSET:
        if groups is None:
            raise ValueError(f"{where}: groups required when inner_caches not supplied")
        caches = build_inner_caches(
            x, np.asarray(groups), n_inner=n_inner, seed=seed, device=device
        )
    if caches is None:
        logger.warning(
            "[fits2091] %s: <2 usable inner group folds — dof-capped GCV fallback", where
        )
        return _gcv_select_capped(x, y_pool, grid=grid, dof_cap=dof_cap, where=where)

    excluded_mask, fold_ntr = _dof_cap_exclusions(caches, grid, dof_cap)
    if excluded_mask.all():
        raise RuntimeError(
            f"{where}: dof_cap={dof_cap} excludes EVERY grid lambda "
            f"(inner_fold_ntr={fold_ntr}, d={d}) — no admissible lambda"
        )
    dev = caches[0]["w"].device
    ytr = torch.as_tensor(np.asarray(y_pool, dtype=np.float64), device=dev)
    rss = fit825._inner_cv_rss_curve(caches, ytr, lams=grid).cpu().numpy()
    masked = np.where(excluded_mask, np.inf, rss)
    best_idx = int(np.argmin(masked))  # first minimum — matches torch.argmin tie-break
    sel = LambdaSelection(
        best_lambda=float(grid[best_idx]),
        selector="inner-group-cv",
        grid=[float(v) for v in grid],
        rss_curve=[float(v) for v in rss],
        gcv_curve=None,
        excluded_lambdas=[float(grid[i]) for i in np.flatnonzero(excluded_mask)],
        dof_cap=dof_cap,
        n_inner_folds_used=len(caches),
        inner_fold_ntr=fold_ntr,
        n_train=ntr,
        d=d,
        where=where,
    )
    _log_selector(sel.selector, sel.best_lambda)
    return sel


# ── final fit at the selected lambda (the #1739 cores, 1-element grid) ────────
def fit_predict_at_lambda(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    lam: float,
    *,
    device: str = "cpu",
) -> np.ndarray:
    """One 1-element-grid ``ridge_layer_batched_auto`` call (primal/dual routed).

    With a single-lambda grid nothing is SELECTED, so the cores' plain GCV is
    inert (the #1887 1-element-forced-grid carve-out); the #825 discipline
    lives entirely in :func:`select_lambda`. Accepts (n, d) 2-D slices or
    (L, n, d) layer-stacked 3-D arrays; returns matching-rank predictions.
    """
    fits1739 = _fits1739()
    x = np.asarray(x_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    xe = np.asarray(x_eval, dtype=np.float64)
    squeeze = x.ndim == 2
    if squeeze:
        x, y, xe = x[None], y[None], xe[None]
    pred = fits1739.ridge_layer_batched_auto(
        x, y, xe, lambdas=np.asarray([float(lam)], dtype=np.float64), device=device
    )
    return pred[0] if squeeze else pred


# ── the per-(pool x regime) wrapper call (3 layers = 3 solves) ────────────────
@dataclass
class FitResult:
    """One wrapper call's artifact: per-layer selection + diagnostics + preds."""

    fit_id: str
    layers: list[int]
    n_train: int
    n_eval: int
    d: int
    d_out: int
    per_layer: list[dict]
    elapsed_s: float
    predictions: np.ndarray | None = field(default=None, repr=False)  # (L, m, d_out)

    def to_json(self) -> dict:
        d = asdict(self)
        d.pop("predictions")  # arrays persist separately (unit C, npz)
        return d


def fit_pool_regime(
    x_pool: np.ndarray,
    y_pool: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    groups: np.ndarray,
    *,
    layers: Sequence[int] = LAYERS,
    fit_id: str = "",
    lambdas: Sequence[float] = LAMBDA_GRID,
    n_inner: int = N_INNER_FOLDS,
    seed: int = SEED,
    dof_cap: float | None = DOF_CAP,
    device: str = "cpu",
    inner_caches: list[list[dict] | None] | None = None,
    knn_ks: tuple[int, ...] = KNN_KS,
    eval_groups: Iterable[str] | None = None,
) -> FitResult:
    """One (pool x regime) fit across the frozen layer set — the plan's wrapper call.

    Per layer: inner-group-CV lambda selection (dof-capped, refusal-guarded),
    then ONE 1-element-grid core call at the selected lambda; the
    mapping-baselines pair (identity+learned-bias + kNN retrieval, chance
    stated) is computed per layer via ``fits.map_diagnostics``. ``inner_caches``
    (one entry per layer) shares the Y-independent eigh work across regimes.
    """
    x_pool = np.asarray(x_pool)
    y_pool = np.asarray(y_pool)
    x_eval = np.asarray(x_eval)
    y_eval = np.asarray(y_eval)
    n_layers = len(layers)
    assert x_pool.ndim == 3 and x_pool.shape[0] == n_layers, (x_pool.shape, layers)
    assert y_pool.shape[:2] == x_pool.shape[:2], (y_pool.shape, x_pool.shape)
    assert x_eval.ndim == 3 and x_eval.shape[0] == n_layers, (x_eval.shape, layers)
    assert y_eval.shape[:2] == x_eval.shape[:2], (y_eval.shape, x_eval.shape)
    groups = np.asarray(groups)
    assert groups.shape[0] == x_pool.shape[1], (groups.shape, x_pool.shape)
    if eval_groups is not None:
        assert_group_disjoint(groups, eval_groups, where=fit_id or "fit_pool_regime")

    fits1739 = _fits1739()
    t0 = time.time()
    if inner_caches is None:
        inner_caches = [
            build_inner_caches(x_pool[li], groups, n_inner=n_inner, seed=seed, device=None)
            for li in range(n_layers)
        ]
    assert len(inner_caches) == n_layers, (len(inner_caches), n_layers)

    preds = np.empty((n_layers, x_eval.shape[1], y_eval.shape[2]), dtype=np.float64)
    selections: list[LambdaSelection] = []
    for li, layer in enumerate(layers):
        sel = select_lambda(
            x_pool[li],
            y_pool[li],
            groups,
            lambdas=lambdas,
            n_inner=n_inner,
            seed=seed,
            dof_cap=dof_cap,
            inner_caches=inner_caches[li],
            where=f"{fit_id}/L{layer}",
        )
        selections.append(sel)
        preds[li] = fit_predict_at_lambda(
            x_pool[li], y_pool[li], x_eval[li], sel.best_lambda, device=device
        )

    diag = fits1739.map_diagnostics(
        preds, x_eval, y_eval, x_pool, y_pool, knn_ks=tuple(int(k) for k in knn_ks)
    )
    per_layer = []
    for li, layer in enumerate(layers):
        row = dict(diag["per_layer"][li])
        row["layer"] = int(layer)
        row["selection"] = selections[li].to_json()
        per_layer.append(row)
    return FitResult(
        fit_id=fit_id,
        layers=[int(v) for v in layers],
        n_train=int(x_pool.shape[1]),
        n_eval=int(x_eval.shape[1]),
        d=int(x_pool.shape[2]),
        d_out=int(y_pool.shape[2]),
        per_layer=per_layer,
        elapsed_s=time.time() - t0,
        predictions=preds,
    )


# ── synthetic pilot (the G3 production call path at toy shape) ────────────────
def synthetic_cell(
    *,
    seed: int = 0,
    n_groups: int = 40,
    group_size: int = 4,
    n_eval_groups: int = 10,
    d: int = 24,
    d_out: int = 24,
    n_layers: int = 3,
) -> dict:
    """Grouped synthetic pool->eval transfer cell (group-disjoint by construction)."""
    rng = np.random.default_rng(seed)
    n = n_groups * group_size
    groups_all = np.repeat([f"g{gi:03d}" for gi in range(n_groups)], group_size)
    x = np.empty((n_layers, n, d))
    y = np.empty((n_layers, n, d_out))
    for li in range(n_layers):
        w_true = rng.normal(size=(d, d_out)) / np.sqrt(d)
        g_eff = rng.normal(size=(n_groups, d))
        xi = rng.normal(size=(n, d)) + np.repeat(g_eff, group_size, axis=0)
        yi = xi @ w_true + 0.5 * rng.normal(size=(n, d_out))
        x[li], y[li] = xi, yi
    eval_gs = {f"g{gi:03d}" for gi in range(n_eval_groups)}
    ev = np.array([g in eval_gs for g in groups_all])
    return {
        "x_pool": x[:, ~ev],
        "y_pool": y[:, ~ev],
        "x_eval": x[:, ev],
        "y_eval": y[:, ev],
        "groups_pool": groups_all[~ev],
        "groups_eval": groups_all[ev],
    }


def run_pilot_synthetic(args: argparse.Namespace) -> dict:
    """Drive ``fit_pool_regime`` end-to-end on a synthetic cell (G3 call path).

    Smoke-output discipline: writes under a scratch root (default
    ``/tmp/issue-2091-smoke/``), NEVER ``eval_results/``.
    """
    cell = synthetic_cell(seed=args.seed)
    t0 = time.time()
    res = fit_pool_regime(
        cell["x_pool"],
        cell["y_pool"],
        cell["x_eval"],
        cell["y_eval"],
        cell["groups_pool"],
        fit_id="pilot-synthetic/regime=greedy",
        device=args.device,
        eval_groups=cell["groups_eval"],
    )
    digest = {
        "mode": "pilot-synthetic",
        "elapsed_s": round(time.time() - t0, 3),
        "fit": res.to_json(),
        "selector_log": SELECTOR_LOG,
    }
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "fits_pilot_synthetic.json"
    out_path.write_text(json.dumps(digest, indent=1) + "\n")
    for row in res.per_layer:
        sel = row["selection"]
        acc1 = row["knn"]["cosine"]["acc_at_k"][1]  # knn_retrieval keys acc_at_k by INT k
        print(
            f"[fits2091-pilot] L{row['layer']}: lambda={sel['best_lambda']:.4g} "
            f"selector={sel['selector']} excluded={len(sel['excluded_lambdas'])} "
            f"r2_map={row['r2_map']:.4f} r2_ib={row['r2_identity_bias']:.4f} "
            f"knn_cos_acc@1={acc1:.3f}",
            flush=True,
        )
    print(f"[fits2091-pilot] wrote {out_path} elapsed={digest['elapsed_s']}s", flush=True)
    return digest


# ── import-check (Axis 1 (a): resolves every deferred import) ─────────────────
def _import_check() -> int:
    """Resolve every deferred/lazy import this module reaches on its real paths."""
    import torch  # noqa: F401

    fit825 = _fit825()
    for name in (
        "_prep_inner_lambda",
        "_inner_cv_rss_curve",
        "_prep_fold",
        "_refuse_unguarded_gcv",
        "_validate_lambda_grid",
        "_as_f64_on",
        "heldout_r2_sweep",
    ):
        assert hasattr(fit825, name), name
    grid = np.asarray(fit825.LAMBDAS, dtype=np.float64)
    assert np.allclose(grid, np.asarray(LAMBDA_GRID)), "LAMBDA_GRID drifted from #825 LAMBDAS"
    assert fit825.GCV_DOF_CAP == DOF_CAP, (fit825.GCV_DOF_CAP, DOF_CAP)
    fits1739 = _fits1739()
    for name in ("ridge_layer_batched_auto", "ridge_gcv_predict_per_target", "map_diagnostics"):
        assert hasattr(fits1739, name), name
    from explore_persona_space.analysis.mapping_baselines import (  # noqa: F401
        identity_bias_predict,
        knn_retrieval,
    )
    from explore_persona_space.experiments.issue_1739.judging import (  # noqa: F401
        alias_correct,
        three_way_classify,
    )

    print("[fits2091] import-check OK (deferred imports resolved; grid/cap pinned to #825)")
    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", choices=("import-check", "pilot-synthetic"), required=True)
    ap.add_argument("--device", default="cpu", help="torch device for the core calls")
    ap.add_argument("--seed", type=int, default=0, help="synthetic-cell seed (pilot mode)")
    ap.add_argument(
        "--out",
        default="/tmp/issue-2091-smoke",
        help="pilot output DIR (scratch — never eval_results/)",
    )
    args = ap.parse_args(argv)
    if args.out.endswith((".json", ".jsonl")):  # out-arg FILE-vs-DIR kind guard (#1776)
        ap.error(f"--out expects a DIRECTORY, got a file-shaped path: {args.out}")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parse_args(argv)
    if args.mode == "import-check":
        return _import_check()
    if args.mode == "pilot-synthetic":
        run_pilot_synthetic(args)
        return 0
    raise SystemExit(f"unknown mode {args.mode!r}")


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit BEFORE C-extension finalize teardown (gotchas.md)
