"""Issue #1902 follow-up — mapping-similarity LADDER over the cross-stage transfer battery.

The parent #1902 P4 transfer battery (``scripts/issue1902_fits.py::run_xfer_unit``)
measured four rungs of the cross-stage map T(i->j): ``direct`` (apply the source
stage's own context->answer map f_ii to the target stage's contexts), the two-sided
general-linear rung ``gl`` (B'(f_ii(A u_j))), the two-sided orthogonal rung ``orth``,
and ``fixedtext``. That leaves the ISOLATING intermediate rungs unmeasured, so the
base->SFT rewrite cannot be localized (constant shift vs gain vs one-sided vs
two-sided coordinate change).

This round adds the seven missing rungs at layer* = 31, single-turn context arm, over
the 12 ordered stage pairs and the parent's 6 cluster-grouped folds:

1. ``ctx_offset``   f_ii(u_j - dx),          dx = mean_tr(u_j) - mean_tr(u_i)
2. ``ans_offset``   f_ii(u_j) + dy,          dy = mean_tr(w_jj) - mean_tr(w_ii)
3. ``bias_refit``   f_ii(u_j) + b*,          b* = mean_tr(w_jj - f_ii(u_j))
4. ``scale_alpha``  a*f_ii(u_j) + b*,        a by 1-D lstsq on train-centered preds
5. ``rot_ans_only`` R(f_ii(u_j) - mu_p) + mu_y,  R = Procrustes(preds_tr -> w_jj[tr])
6. ``gl_ctx_only``  f_ii(A u_j),             A = ridge(u_j[tr] -> u_i[tr])
7. ``gl_ans_only``  B'(f_ii(u_j)),           B' = ridge(w_ii[tr] -> w_ji[tr])

plus three PARITY re-derivations of the parent's own rungs from the same store —
``direct``, ``orth_2sided`` (= parent ``orth``), ``gl_2sided`` (= parent ``gl``) — and
the target's diagonal ``Q_jj`` (the retention denominator), each checked against the
persisted ``eval_results/issue_1902/transfer/transfer_matrix.json``. Parity is the RIG
GATE: the new rungs are only trustworthy once the re-derived old ones reproduce.

Every correction is fitted on TRAIN folds only and scored fold-HELD-OUT against the
SAME target ``w_jj[ev]`` (matched target), so all ten rungs are directly comparable.
Retention = mode R^2 / Q_jj, the parent's ``retention_gl`` convention.

Reuse (artifact-reuse check (i)): fold masks, the store cache + its row-id asserts,
the corpus index, the orthogonal-Procrustes map, and the per-context SS reduction all
come from ``issue1902_fits`` / ``issue1902_run`` verbatim, so folds / row order /
lambda grid match the parent exactly.

RIDGE SOLVER. The parent's ``ridge_fit_predict_fast_layer_batched`` solves in the DUAL
(n_tr x n_tr) Gram space, which is the right factorization for #779's n < d regime but
the wrong one here: n_tr ~= 13.7k >> d = 4096, so the dual eigh is ~37x the primal's
work and cannot reuse a factorization across targets. ``_factorize``/``_solve`` below
implement the SAME estimator in the primal (d x d) space -- standardize-X on train
stats, center-Y, GCV over ``logspace(-2, 4, 13)`` selected per (X, Y) pair, uncentered
predictions, float64 -- which is algebraically identical (G and X'X share their nonzero
spectrum; the dual's null-space directions are annihilated by K_ev) and lets ONE
factorization serve every target that shares its design matrix. ``--verify-ridge``
gates it against the parent solver on >= 3 slices at production shape (tol 1e-4, the
parent's own ``PARITY_TOL``); the end-to-end parity legs above are the second check.

Content hygiene: corpus / rollout text never enters logs or outputs -- ids, indices,
counts and float summaries only.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_SCRIPTS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPTS_DIR.parent
for _p in (str(_SCRIPTS_DIR), str(_PROJECT_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# load_dotenv() BEFORE any heavy import so the shared-VM thread caps (#847) bind
# in-process (tests/test_shared_vm_thread_caps.py, the #1146 predicate).
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue1902_common as C  # noqa: E402
import issue1902_fits as F  # noqa: E402
import issue1902_run as R  # noqa: E402

logger = R.logger

LADDER_RECIPE_VERSION = "issue1902-ladder-v1"
FOLLOWUP_LABEL = "ladder_modes"

LAYER_STAR = 31
CORPUS = C.CORPUS_SINGLE
ARM = F.ARM_CTX
LAMBDAS = np.logspace(-2, 4, 13)  # fit_h.ridge_fit_predict_fast default grid

# Ladder rungs, ORDERED by constraint strength (fewest free parameters first).
LADDER_MODES: tuple[str, ...] = (
    "direct",
    "ctx_offset",
    "ans_offset",
    "bias_refit",
    "scale_alpha",
    "rot_ans_only",
    "gl_ctx_only",
    "gl_ans_only",
    "orth_2sided",
    "gl_2sided",
)
# Rungs that re-derive a parent transfer_matrix.json mode (the rig gate).
PARITY_MODE_MAP = {"direct": "direct", "orth_2sided": "orth", "gl_2sided": "gl"}
PARITY_TOL_R2 = 0.01  # brief: investigate a deviation > ~0.01
RIDGE_PARITY_TOL = F.PARITY_TOL  # 1e-4, the parent's slow-vs-fast tolerance

# Retention reference lines on the figure.
RETENTION_REFLINES = (0.8, 0.5, 0.0)

ADJACENT = F.ADJACENT_PAIRS  # (B,S), (S,D), (D,R)

DEFAULT_OUT_ROOT = Path(
    os.environ.get(
        "EPM_ISSUE1902_LADDER_OUT_ROOT",
        f"/mnt/eps-data/{os.environ.get('USER', 'thomasjiralerspong')}/issue1902_ladder",
    )
)
REPO_EVAL_DIR = R.PROJECT_ROOT / "eval_results" / "issue_1902" / "followup_ladder"
REPO_FIG_DIR = R.PROJECT_ROOT / "figures" / "issue_1902"
PARENT_XFER_JSON = (
    R.PROJECT_ROOT / "eval_results" / "issue_1902" / "transfer" / "transfer_matrix.json"
)


def ordered_pairs(ckpts: list[str]) -> list[tuple[str, str]]:
    return [(i, j) for i in ckpts for j in ckpts if i != j]


# ── input staging (self-contained; idempotent) ────────────────────────────────


def _needed_store_files() -> list[str]:
    """Store-relative paths this round reads: the layer* ctx + answer cells of the
    single-turn corpus, plus the ctx row_index manifests ``CorpusIndex`` hard-reads."""
    rel: list[str] = []
    for m in C.CKPTS:
        rel.append(C.ctx_store_relpath(m, CORPUS, LAYER_STAR))
        rel.append(C.cell_row_index_relpath(m, C.CTX_SOURCE, CORPUS))
        for src in C.CKPTS:
            rel.append(C.answer_store_relpath(m, src, CORPUS, LAYER_STAR))
    return rel


def stage_inputs(out_root: Path) -> dict[str, Any]:
    """Stage the layer* single-turn slices + the intersection manifest from HF.

    ``hub.stage_hub_file`` takes an EXACT per-file target (no #1774 mirror-root
    trap), is atomic + ``retry_transient``-wrapped, and skips already-present files,
    so a partial stage heals idempotently. One resolved revision covers every file
    (#833 scoped-listing recipe)."""
    from concurrent.futures import ThreadPoolExecutor

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    store = R._store_root(out_root)
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = hub.retry_transient(
        lambda: api.repo_info(C.HF_DATA_REPO, repo_type="dataset"),
        what=f"repo_info({C.HF_DATA_REPO})",
    )
    revision = str(info.sha)

    manifest_target = out_root / "gen" / "intersection_manifest.json"
    jobs: list[tuple[str, Path]] = [
        (f"{C.EVAL_MIRROR_HF_PATH}/gen/intersection_manifest.json", manifest_target)
    ]
    for rel in _needed_store_files():
        jobs.append((f"{C.STORE_HF_PATH}/{rel}", store / rel))
    pending = [(src, tgt) for src, tgt in jobs if not tgt.exists()]
    print(
        f"[ladder] stage: {len(jobs)} inputs, {len(pending)} missing "
        f"(revision={revision[:12]}, dest={out_root})",
        flush=True,
    )
    if pending:
        with ThreadPoolExecutor(max_workers=6) as pool:
            futs = [
                pool.submit(
                    hub.stage_hub_file,
                    C.HF_DATA_REPO,
                    src,
                    tgt,
                    repo_type="dataset",
                    revision=revision,
                )
                for src, tgt in pending
            ]
            for fut in futs:
                fut.result()  # re-raises — fail-loud
    still = [str(tgt) for _, tgt in jobs if not tgt.exists()]
    if still:
        raise FileNotFoundError(f"staging left inputs missing: {still[:4]}")
    nbytes = sum(tgt.stat().st_size for _, tgt in jobs)
    print(
        f"[ladder] stage OK: {len(jobs)} files, {nbytes / 1e9:.2f} GB under {out_root}", flush=True
    )
    return {"revision": revision, "n_files": len(jobs), "bytes": int(nbytes)}


# ── run context (single-corpus slice of the parent FitsContext) ───────────────


class LadderContext(F.FitsContext):
    """Single-corpus, single-layer slice of the parent ``FitsContext``.

    ``__init__`` is replaced (the parent's builds BOTH corpora and discovers every
    captured layer, which would demand staging the whole store) but the data
    accessors used downstream — ``xy`` and ``fold_masks`` — are INHERITED verbatim,
    so row order, the store row-id asserts, the ctx/answer key names and the fold
    assignment are the parent's own code paths."""

    def __init__(self, out_root: Path, ckpts: list[str], *, layer: int = LAYER_STAR):
        self.args = argparse.Namespace(smoke=False)
        self.smoke = False
        self.out_root = out_root
        self.ckpts = ckpts
        self.store = R._store_root(out_root)
        self.eval_dir = out_root / "eval_results" / "issue_1902"
        cache_gb = float(os.environ.get("EPM_ISSUE1902_LADDER_CACHE_GB", "10"))
        self.cache = F.StoreCache(self.store, cap_gb=cache_gb)
        self.corpora = {CORPUS: F.CorpusIndex(out_root, CORPUS, ckpts, self.store)}
        self.layers = [layer]
        self.layer_star = layer
        self.layer_star_p = None
        self.band = [layer]
        self.band_p = []
        self.pilot_timings = []

    def unit_paths(self) -> tuple[Path, Path]:
        return (
            self.eval_dir / FOLLOWUP_LABEL / "units",
            self.eval_dir / FOLLOWUP_LABEL / "percell",
        )


def ladder_regime(ctx: LadderContext) -> dict[str, Any]:
    """Every output-affecting regime key (a mismatch REFUSES the resume, #1333)."""
    return {
        "recipe_version": LADDER_RECIPE_VERSION,
        "layer": ctx.layer_star,
        "corpus": CORPUS,
        "arm": ARM,
        "ckpts": list(ctx.ckpts),
        "modes": list(LADDER_MODES),
        "lambdas": [float(x) for x in LAMBDAS],
        "ridge_impl": "primal-gcv-float64",
        "n_folds": ctx.corpora[CORPUS].n_folds,
    }


# ── primal GCV ridge (algebraically identical to the parent's dual solver) ────


@dataclass
class _Fact:
    """Cached primal factorization of ONE standardized design matrix."""

    xmu: Any  # (d,) train mean
    xsd: Any  # (d,) train population std + 1e-9
    s: Any  # (d,) eigenvalues of Xn' Xn, clamped >= 0
    U: Any  # (d, d) eigenvectors
    Xn: Any  # (n, d) standardized train inputs (needed per new target)
    n: int


@dataclass
class _Beta:
    """Fitted primal map: standardization stats + weights + target mean."""

    xmu: Any
    xsd: Any
    beta: Any  # (d, d_out)
    ymu: Any  # (d_out,)
    lam: float
    dof: float


def _factorize(Xtr: np.ndarray, device: str) -> _Fact:
    import torch

    dev = torch.device(device)
    Xt = torch.as_tensor(np.asarray(Xtr), dtype=torch.float64, device=dev)
    xmu = Xt.mean(dim=0)
    xsd = Xt.std(dim=0, unbiased=False) + 1e-9  # population std (twin parity)
    Xn = (Xt - xmu) / xsd
    S = Xn.T @ Xn
    s, U = torch.linalg.eigh(S)
    return _Fact(xmu=xmu, xsd=xsd, s=torch.clamp(s, min=0.0), U=U, Xn=Xn, n=int(Xn.shape[0]))


def _solve(fact: _Fact, Ytr: np.ndarray, device: str) -> _Beta:
    """GCV-selected ridge for one target against a cached factorization.

    Same selection objective as the parent's dual solver: with ``P = U' Xn' Yc``,
    ``rss(lam) = tot - 2 sum ||P_i||^2/(s_i+lam) + sum s_i ||P_i||^2/(s_i+lam)^2`` and
    ``dof(lam) = sum s_i/(s_i+lam)``, minimizing ``rss / (n - dof)^2`` over ``LAMBDAS``
    — the algebraic image of the dual's ``rss = tot - sum (2f - f^2) |V'Yc|^2`` under
    ``V[:, k] = Xn U[:, i]/sqrt(s_i)`` (zero-eigenvalue dual directions contribute
    nothing and are annihilated by ``K_ev``)."""
    import torch

    dev = torch.device(device)
    Yt = torch.as_tensor(np.asarray(Ytr), dtype=torch.float64, device=dev)
    ymu = Yt.mean(dim=0)
    Yc = Yt - ymu
    P = fact.U.T @ (fact.Xn.T @ Yc)  # (d, d_out)
    normP = (P * P).sum(dim=1)  # (d,)
    tot = float((Yc * Yc).sum())
    s, n = fact.s, fact.n
    best = (float("inf"), None, None, None)
    for lam in LAMBDAS:
        den = s + float(lam)
        rss = tot - 2.0 * float((normP / den).sum()) + float((s * normP / den**2).sum())
        dof = float((s / den).sum())
        denom = (n - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else float("inf")
        if gcv < best[0]:
            best = (gcv, float(lam), dof, den)
    _, lam_star, dof_star, den_star = best
    assert den_star is not None, "GCV selected no lambda (all denominators degenerate)"
    beta = fact.U @ (P / den_star[:, None])  # (d, d_out)
    return _Beta(
        xmu=fact.xmu, xsd=fact.xsd, beta=beta, ymu=ymu, lam=float(lam_star), dof=float(dof_star)
    )


def _apply(b: _Beta, Xev: np.ndarray, device: str, chunk: int = 4096) -> np.ndarray:
    """Apply a fitted primal map to eval inputs, row-chunked (bounded peak)."""
    import torch

    dev = torch.device(device)
    Xt = torch.as_tensor(np.asarray(Xev), dtype=torch.float64, device=dev)
    out = torch.empty((Xt.shape[0], b.beta.shape[1]), dtype=torch.float64, device=dev)
    for a in range(0, Xt.shape[0], chunk):
        blk = Xt[a : a + chunk]
        out[a : a + chunk] = ((blk - b.xmu) / b.xsd) @ b.beta + b.ymu
    return out.cpu().numpy()


def verify_ridge(ctx: LadderContext, device: str, *, fold: int = 0) -> dict[str, Any]:
    """Slice-level gate: primal solver vs the parent ``_batched_ridge`` at PRODUCTION
    shape on one diagonal fit per checkpoint. Tol = the parent's own ``PARITY_TOL``."""
    tr, ev = ctx.fold_masks(CORPUS, fold)
    slices: list[dict[str, Any]] = []
    worst = 0.0
    worst_lam = 0
    for m in ctx.ckpts:
        X, Y = ctx.xy(m, m, CORPUS, ctx.layer_star, ARM)
        fact = _factorize(X[tr], device)
        b = _solve(fact, Y[tr], device)
        mine = _apply(b, X[ev], device)
        theirs, info = F._batched_ridge(
            X[tr][None], Y[tr][None], X[ev][None], device=device, return_info=True
        )
        theirs = theirs[0]
        their_lam = float(info["best_lambda"][0])
        denom = max(float(np.abs(theirs).max()), 1e-12)
        rel = float(np.abs(mine - theirs).max() / denom)
        lam_mismatch = int(their_lam != b.lam)
        worst = max(worst, rel)
        worst_lam += lam_mismatch
        slices.append(
            {
                "ckpt": m,
                "fold": fold,
                "n_tr": int(tr.sum()),
                "n_ev": int(ev.sum()),
                "d": int(X.shape[1]),
                "max_rel": rel,
                "lambda_primal": b.lam,
                "lambda_parent": their_lam,
                "dof_primal": b.dof,
                "dof_parent": float(info["dof"][0]),
            }
        )
        print(
            f"[ladder] verify-ridge {m}: max_rel={rel:.3e} lam={b.lam:g}/{their_lam:g}",
            flush=True,
        )
        del fact, b, mine, theirs
    report = {
        "tol": RIDGE_PARITY_TOL,
        "max_rel_diff": worst,
        "n_lambda_mismatch": worst_lam,
        "slices": slices,
        "pass": bool(worst <= RIDGE_PARITY_TOL and worst_lam == 0 and len(slices) >= 3),
    }
    return report


# ── the ladder unit: one (i -> j, fold) cell ─────────────────────────────────


def _unit_name(i: str, j: str, fold: int) -> str:
    return f"ladder_{i}{j}_f{fold}"


def _unit_done(ctx: LadderContext, i: str, j: str, fold: int, regime: dict) -> bool:
    units_dir, _ = ctx.unit_paths()
    path = units_dir / f"{_unit_name(i, j, fold)}.json"
    if not path.exists():
        return False
    try:
        prev = R._read_json(path)
    except Exception:  # noqa: BLE001 — a truncated unit is simply redone
        return False
    return prev.get("regime") == regime


def run_fold(ctx: LadderContext, device: str, fold: int, pairs: list[tuple[str, str]]) -> int:
    """Compute every pending ladder cell of ONE fold.

    Factorizations are shared the way the algebra allows: the standardized ctx Gram of
    stage m serves BOTH f_mm (target ``w_mm``) and every A_ctx that maps FROM m, and
    the standardized answer Gram of stage i serves every B' that maps w_ii -> w_ji.
    Per-target lambda selection is preserved (each target re-runs GCV on the shared
    spectrum), so this is a factorization reuse, not a pooled fit."""
    import torch

    regime = ladder_regime(ctx)
    idx = ctx.corpora[CORPUS]
    tr, ev = ctx.fold_masks(CORPUS, fold)
    n_tr, n_ev = int(tr.sum()), int(ev.sum())
    if n_ev < 2 or n_tr < 2:
        raise RuntimeError(f"fold {fold} degenerate: n_tr={n_tr} n_ev={n_ev}")
    pending = [(i, j) for i, j in pairs if not _unit_done(ctx, i, j, fold, regime)]
    if not pending:
        print(f"[ladder] fold {fold}: all {len(pairs)} pairs done — skip", flush=True)
        return 0
    layer = ctx.layer_star
    sources = sorted({i for i, _ in pending})
    targets = sorted({j for _, j in pending})

    # Row-aligned slices (loaded through the parent accessors: row-id asserted).
    u: dict[str, np.ndarray] = {}
    w_diag: dict[str, np.ndarray] = {}
    for m in sorted(set(sources) | set(targets)):
        u[m], w_diag[m] = ctx.xy(m, m, CORPUS, layer, ARM)

    # ---- stage 1: context-side maps + the diagonal f_mm (one ctx Gram alive) ----
    beta_f: dict[str, _Beta] = {}
    q_diag: dict[str, dict[str, float]] = {}
    gl_ev: dict[tuple[str, str], np.ndarray] = {}
    orth_ev: dict[tuple[str, str], np.ndarray] = {}
    ctx_hosts = sorted(set(sources) | set(targets))
    for m in ctx_hosts:
        t0 = time.time()
        fact = _factorize(u[m][tr], device)
        if m in sources:
            beta_f[m] = _solve(fact, w_diag[m][tr], device)
        if m in targets:
            # Q_mm: the target's OWN diagonal at layer* — the retention denominator
            # (parent convention retention_gl = R2_mode / Q_jj) and a parity leg.
            b_diag = beta_f.get(m) or _solve(fact, w_diag[m][tr], device)
            pred = _apply(b_diag, u[m][ev], device)
            res, tot, cos = F._per_ctx_ss(pred, w_diag[m][ev], w_diag[m][tr].mean(0))
            q_diag[m] = {
                "ss_res": float(res.sum()),
                "ss_tot": float(tot.sum()),
                "r2": F._pooled_r2(float(res.sum()), float(tot.sum())),
                "cos_mean": float(np.mean(cos)),
                "lambda": b_diag.lam,
                "dof": b_diag.dof,
            }
            del b_diag, pred, res, tot, cos
        # A_ctx / R_ctx map the CONTEXT cloud of m into stage i's context space; the
        # eval-side application happens here so the factorization can be dropped.
        for i in sources:
            if i == m or (i, m) not in pending:
                continue
            b_a = _solve(fact, u[i][tr], device)
            gl_ev[(i, m)] = _apply(b_a, u[m][ev], device)
            del b_a
            R_ctx, mu_from, mu_to = F._orth_map(u[m][tr], u[i][tr], device)
            orth_ev[(i, m)] = (u[m][ev] - mu_from) @ R_ctx + mu_to
            del R_ctx, mu_from, mu_to
        del fact
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        print(
            f"[ladder] fold {fold} ctx-host {m}: stage-1 maps in {time.time() - t0:.1f}s",
            flush=True,
        )

    # ---- stage 2: apply f_ii + answer-side maps (one answer Gram alive) --------
    units_dir, percell_dir = ctx.unit_paths()
    percell_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for i in sources:
        js = [j for src, j in pending if src == i]
        if not js:
            continue
        ans_fact = _factorize(w_diag[i][tr], device)
        for j in js:
            t0 = time.time()
            bf = beta_f[i]
            f_dir_ev = _apply(bf, u[j][ev], device)
            f_dir_tr = _apply(bf, u[j][tr], device)
            dx = u[j][tr].mean(0) - u[i][tr].mean(0)
            preds: dict[str, np.ndarray] = {"direct": f_dir_ev}
            preds["ctx_offset"] = _apply(bf, u[j][ev] - dx, device)
            dy = w_diag[j][tr].mean(0) - w_diag[i][tr].mean(0)
            preds["ans_offset"] = f_dir_ev + dy
            b_star = (w_diag[j][tr] - f_dir_tr).mean(0)
            preds["bias_refit"] = f_dir_ev + b_star
            # scale_alpha: single gain on train-centered predictions + matching intercept.
            p_mu = f_dir_tr.mean(0)
            y_mu = w_diag[j][tr].mean(0)
            pc = f_dir_tr - p_mu
            yc = w_diag[j][tr] - y_mu
            denom_a = float((pc * pc).sum())
            alpha = float((pc * yc).sum() / denom_a) if denom_a > 0 else float("nan")
            preds["scale_alpha"] = alpha * (f_dir_ev - p_mu) + y_mu
            del pc, yc
            # rot_ans_only: answer-side orthogonal map from predictions to targets.
            R5, mu_p, mu_y = F._orth_map(f_dir_tr, w_diag[j][tr], device)
            preds["rot_ans_only"] = (f_dir_ev - mu_p) @ R5 + mu_y
            del R5, mu_p, mu_y
            preds["gl_ctx_only"] = _apply(bf, gl_ev[(i, j)], device)
            # Answer-side general-linear map B' fitted on SAME-answer-text pairs
            # w_ii[tr] -> w_ji[tr] (the parent A_ans convention: fit the
            # target<-source direction directly, never invert a learned map).
            w_ji = ctx.cache.answer(j, i, CORPUS, layer, idx.ids)
            b_b = _solve(ans_fact, w_ji[tr], device)
            preds["gl_ans_only"] = _apply(b_b, f_dir_ev, device)
            preds["gl_2sided"] = _apply(b_b, preds["gl_ctx_only"], device)
            R_ans, mu_wii, mu_wji = F._orth_map(w_diag[i][tr], w_ji[tr], device)
            f_orth = _apply(bf, orth_ev[(i, j)], device)
            preds["orth_2sided"] = (f_orth - mu_wii) @ R_ans + mu_wji
            del R_ans, mu_wii, mu_wji, f_orth

            y_tgt = w_diag[j][ev]
            y_mean = w_diag[j][tr].mean(0)
            per_mode: dict[str, dict[str, float]] = {}
            arrays: dict[str, np.ndarray] = {"row_idx": np.flatnonzero(ev)}
            tot_ref: np.ndarray | None = None
            for mode in LADDER_MODES:
                res, tot, cos = F._per_ctx_ss(preds[mode], y_tgt, y_mean)
                tot_ref = tot
                per_mode[mode] = {
                    "ss_res": float(res.sum()),
                    "ss_tot": float(tot.sum()),
                    "r2": F._pooled_r2(float(res.sum()), float(tot.sum())),
                    "cos_mean": float(np.mean(cos)),
                }
                arrays[f"ss_res_{mode}"] = res
            assert tot_ref is not None
            arrays["ss_tot"] = tot_ref
            F._savez_atomic(percell_dir / f"{_unit_name(i, j, fold)}.npz", **arrays)
            payload = {
                "regime": regime,
                "metadata": R._metadata(),
                "pair": f"{i}->{j}",
                "i": i,
                "j": j,
                "fold": fold,
                "layer": layer,
                "corpus": CORPUS,
                "arm": ARM,
                "n_tr": n_tr,
                "n_ev": n_ev,
                "d": int(u[j].shape[1]),
                "modes": per_mode,
                "q_jj": q_diag[j],
                "alpha_scale": alpha,
                "lambda_f_ii": bf.lam,
                "dof_f_ii": bf.dof,
                "lambda_b_ans": b_b.lam,
                "elapsed_s": time.time() - t0,
            }
            R._write_json_atomic(units_dir / f"{_unit_name(i, j, fold)}.json", payload)
            written += 1
            print(
                f"[ladder] unit {written}/{len(pending)} {i}->{j} fold={fold} "
                f"n_tr={n_tr} n_ev={n_ev} elapsed={payload['elapsed_s']:.1f}s "
                f"direct={per_mode['direct']['r2']:.4f} gl2={per_mode['gl_2sided']['r2']:.4f}",
                flush=True,
            )
            del preds, f_dir_ev, f_dir_tr, b_b, w_ji
            gl_ev.pop((i, j), None)
            orth_ev.pop((i, j), None)
        del ans_fact
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return written


# ── finalize: pool folds, parity gate, emit the result JSON ───────────────────


def _parent_matrix() -> dict[str, Any]:
    if not PARENT_XFER_JSON.exists():
        raise FileNotFoundError(
            f"parent transfer matrix absent at {PARENT_XFER_JSON} — the parity gate "
            "cannot run without it"
        )
    return R._read_json(PARENT_XFER_JSON)


def finalize(ctx: LadderContext, pairs: list[tuple[str, str]], stage_info: dict | None) -> dict:
    regime = ladder_regime(ctx)
    units_dir, _ = ctx.unit_paths()
    n_folds = ctx.corpora[CORPUS].n_folds
    parent = _parent_matrix()
    if int(parent.get("layer_star", -1)) != ctx.layer_star:
        raise RuntimeError(
            f"parent layer_star {parent.get('layer_star')} != ladder layer {ctx.layer_star}"
        )

    out_pairs: dict[str, Any] = {}
    parity_rows: list[dict[str, Any]] = []
    for i, j in pairs:
        units = []
        for k in range(n_folds):
            path = units_dir / f"{_unit_name(i, j, k)}.json"
            if not path.exists():
                raise FileNotFoundError(f"missing ladder unit {path} — run --phase run first")
            unit = R._read_json(path)
            if unit.get("regime") != regime:
                raise RuntimeError(f"regime mismatch in {path} — stale unit, re-run the fold")
            units.append(unit)
        q_res = float(np.sum([u["q_jj"]["ss_res"] for u in units]))
        q_tot = float(np.sum([u["q_jj"]["ss_tot"] for u in units]))
        q_jj = F._pooled_r2(q_res, q_tot)
        rec: dict[str, Any] = {
            "q_jj_at_star": q_jj,
            "n_tr_per_fold": [u["n_tr"] for u in units],
            "n_ev_per_fold": [u["n_ev"] for u in units],
            "lambda_f_ii_per_fold": [u["lambda_f_ii"] for u in units],
            "alpha_scale_per_fold": [u["alpha_scale"] for u in units],
            "r2": {},
            "r2_per_fold": {},
            "retention": {},
            "retention_per_fold": {},
            "cos_mean_per_fold": {},
        }
        for mode in LADDER_MODES:
            res = float(np.nansum([u["modes"][mode]["ss_res"] for u in units]))
            tot = float(np.nansum([u["modes"][mode]["ss_tot"] for u in units]))
            r2 = F._pooled_r2(res, tot)
            per_fold = [u["modes"][mode]["r2"] for u in units]
            rec["r2"][mode] = r2
            rec["r2_per_fold"][mode] = per_fold
            rec["cos_mean_per_fold"][mode] = [u["modes"][mode]["cos_mean"] for u in units]
            rec["retention"][mode] = (
                r2 / q_jj if q_jj and np.isfinite(q_jj) and q_jj > 0 else float("nan")
            )
            rec["retention_per_fold"][mode] = [
                (
                    u["modes"][mode]["r2"] / u["q_jj"]["r2"]
                    if u["q_jj"]["r2"] and np.isfinite(u["q_jj"]["r2"]) and u["q_jj"]["r2"] > 0
                    else float("nan")
                )
                for u in units
            ]
        out_pairs[f"{i}->{j}"] = rec

        pj = parent["pairs"][f"{i}->{j}"]
        for mode, parent_mode in PARITY_MODE_MAP.items():
            dev = abs(rec["r2"][mode] - float(pj["r2"][parent_mode]))
            parity_rows.append(
                {
                    "pair": f"{i}->{j}",
                    "leg": f"r2:{mode}",
                    "parent_mode": parent_mode,
                    "ladder": rec["r2"][mode],
                    "parent": float(pj["r2"][parent_mode]),
                    "abs_dev": dev,
                }
            )
        dev_q = abs(q_jj - float(pj["q_jj_at_star"]))
        parity_rows.append(
            {
                "pair": f"{i}->{j}",
                "leg": "q_jj_at_star",
                "parent_mode": "q_jj_at_star",
                "ladder": q_jj,
                "parent": float(pj["q_jj_at_star"]),
                "abs_dev": dev_q,
            }
        )
        parent_lams = pj.get("lambda_star_center") or []
        if len(parent_lams) != len(rec["lambda_f_ii_per_fold"]):
            # Absent/short parent list must fail the leg loudly, never vacuously pass
            # (zip would truncate and score 0 mismatches on an empty parent list).
            mism = len(rec["lambda_f_ii_per_fold"])
        else:
            mism = sum(
                1
                for a, b in zip(rec["lambda_f_ii_per_fold"], parent_lams, strict=True)
                if float(a) != float(b)
            )
        parity_rows.append(
            {
                "pair": f"{i}->{j}",
                "leg": "lambda_f_ii_per_fold",
                "parent_mode": "lambda_star_center",
                "ladder": rec["lambda_f_ii_per_fold"],
                "parent": parent_lams,
                "abs_dev": float(mism),
            }
        )

    r2_devs = [r["abs_dev"] for r in parity_rows if r["leg"].startswith("r2:")]
    q_devs = [r["abs_dev"] for r in parity_rows if r["leg"] == "q_jj_at_star"]
    lam_mism = [r["abs_dev"] for r in parity_rows if r["leg"] == "lambda_f_ii_per_fold"]
    parity = {
        "tol_abs_r2": PARITY_TOL_R2,
        "max_abs_dev_r2": max(r2_devs) if r2_devs else float("nan"),
        "max_abs_dev_q_jj": max(q_devs) if q_devs else float("nan"),
        "n_lambda_mismatch_folds": int(sum(lam_mism)),
        "rows": parity_rows,
        "pass": bool(
            r2_devs
            and max(r2_devs) <= PARITY_TOL_R2
            and max(q_devs) <= PARITY_TOL_R2
            and sum(lam_mism) == 0
        ),
        "parent_matrix": str(PARENT_XFER_JSON.relative_to(R.PROJECT_ROOT)),
        "parent_metadata_git_sha": (parent.get("metadata") or {}).get("git_sha"),
    }

    ridge_gate_path = ctx.eval_dir / FOLLOWUP_LABEL / "ridge_parity.json"
    ridge_gate = R._read_json(ridge_gate_path) if ridge_gate_path.exists() else None
    out = {
        "metadata": R._metadata(),
        "followup_label": FOLLOWUP_LABEL,
        "recipe_version": LADDER_RECIPE_VERSION,
        "regime": regime,
        "layer_star": ctx.layer_star,
        "corpus": CORPUS,
        "arm": ARM,
        "modes_ordered": list(LADDER_MODES),
        "mode_definitions": {
            "direct": "f_ii(u_j) — the source stage's own map applied to target contexts",
            "ctx_offset": "f_ii(u_j - dx), dx = mean_tr(u_j) - mean_tr(u_i)",
            "ans_offset": "f_ii(u_j) + dy, dy = mean_tr(w_jj) - mean_tr(w_ii)",
            "bias_refit": "f_ii(u_j) + b*, b* = mean_tr(w_jj - f_ii(u_j))",
            "scale_alpha": "a*(f_ii(u_j) - mean_tr) + mean_tr(w_jj), a by 1-D lstsq",
            "rot_ans_only": "Procrustes(preds_tr -> w_jj[tr]) applied to f_ii(u_j)",
            "gl_ctx_only": "f_ii(A u_j), A = ridge(u_j[tr] -> u_i[tr])",
            "gl_ans_only": "B'(f_ii(u_j)), B' = ridge(w_ii[tr] -> w_ji[tr])",
            "orth_2sided": "parent `orth` re-derived: R_ans(f_ii(R_ctx u_j))",
            "gl_2sided": "parent `gl` re-derived: B'(f_ii(A u_j))",
        },
        "retention_definition": "pooled mode R2 / pooled Q_jj at layer* (parent retention_gl)",
        "adjacent_transitions": [f"{a}->{b}" for a, b in ADJACENT],
        "n_folds": n_folds,
        "n_train_min": int(min(min(r["n_tr_per_fold"]) for r in out_pairs.values())),
        "n_eval_min": int(min(min(r["n_ev_per_fold"]) for r in out_pairs.values())),
        "d": None,  # filled from a unit below (never a hardcoded literal)
        "estimator_validity": (
            "every fitted correction is well-posed: n_tr >= "
            f"{int(min(min(r['n_tr_per_fold']) for r in out_pairs.values()))} > d "
            "(recorded in `d`); offsets/gains are means and a 1-D lstsq; the "
            "orthogonal maps are closed-form."
        ),
        "store_revision": (stage_info or {}).get("revision"),
        "ridge_parity_gate": ridge_gate,
        "parity_gate": parity,
        "pairs": out_pairs,
    }
    # d recorded from a unit (avoid the literal above going stale).
    any_unit = R._read_json(units_dir / f"{_unit_name(*pairs[0], 0)}.json")
    out["d"] = int(any_unit["d"])

    for dest in (
        ctx.eval_dir / FOLLOWUP_LABEL / "ladder_modes.json",
        REPO_EVAL_DIR / "ladder_modes.json",
    ):
        R._write_json_atomic(dest, out)
    print(
        f"[ladder] finalize: parity pass={parity['pass']} "
        f"max|dev| R2={parity['max_abs_dev_r2']:.2e} Q={parity['max_abs_dev_q_jj']:.2e} "
        f"lambda mismatches={parity['n_lambda_mismatch_folds']} -> "
        f"{REPO_EVAL_DIR / 'ladder_modes.json'}",
        flush=True,
    )
    return out


# ── figure ───────────────────────────────────────────────────────────────────


def make_figure(result: dict) -> Path:
    """Retention vs ladder rung, one line per ADJACENT stage transition."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    modes = list(result["modes_ordered"])
    xs = np.arange(len(modes))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    # The two rungs whose correction is estimated on the CONTEXT clouds alone (they
    # never see the target stage's answer summaries) — every transition dips there,
    # which is the localization read, so mark them rather than leave the dips bare.
    ctx_only = [k for k, m in enumerate(modes) if m in ("ctx_offset", "gl_ctx_only")]
    for k in ctx_only:
        ax.axvspan(k - 0.42, k + 0.42, color=paper_palette_role("neutral"), alpha=0.10, zorder=0)
    for y in RETENTION_REFLINES:
        ax.axhline(
            y,
            color=paper_palette_role("neutral"),
            lw=0.9,
            ls="--" if y else "-",
            alpha=0.7 if y else 0.9,
            zorder=1,
        )
        ax.annotate(
            f"{y:g}",
            xy=(len(modes) - 0.75, y),
            fontsize=7,
            color=paper_palette_role("neutral"),
            va="bottom",
            ha="right",
        )
    roles = ("primary", "baseline", "control", "accent")
    for k, (a, b) in enumerate(ADJACENT):
        key = f"{a}->{b}"
        rec = result["pairs"].get(key)
        if rec is None:
            continue
        ys = [rec["retention"][m] for m in modes]
        ax.plot(
            xs,
            ys,
            marker="o",
            ms=4.5,
            lw=1.6,
            color=paper_palette_role(roles[k % len(roles)]),
            label=f"{key}  (Q$_{{jj}}$={rec['q_jj_at_star']:.2f})",
            zorder=3,
        )
    ax.set_xticks(xs)
    ax.set_xticklabels([m.replace("_", "\n") for m in modes], fontsize=7.5)
    ax.set_xlabel("mapping-similarity ladder rung (fewest free parameters first)")
    ax.set_ylabel("retention  =  mode $R^2$ / $Q_{jj}$")
    ax.set_xlim(-0.6, len(modes) - 0.4)
    if ctx_only:
        ax.annotate(
            "shaded: correction fitted on\ncontext clouds only",
            xy=(ctx_only[-1], ax.get_ylim()[0]),
            xytext=(ctx_only[-1] + 0.15, ax.get_ylim()[0] + 0.06),
            fontsize=7,
            color=paper_palette_role("neutral"),
            ha="left",
            va="bottom",
        )
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title(
        f"Cross-stage transfer retention along the mapping-similarity ladder\n"
        f"layer {result['layer_star']}, {result['corpus']}-turn context arm, "
        f"{result['n_folds']} cluster-grouped folds",
        fontsize=9,
    )
    fig.tight_layout()
    paths = savefig_paper(fig, "ladder_modes", dir=REPO_FIG_DIR, formats=("png",))
    plt.close(fig)
    print(f"[ladder] figure -> {paths}", flush=True)
    return paths["png"]


# ── entrypoint ───────────────────────────────────────────────────────────────


def _import_check() -> None:
    """Resolve every deferred import on the REAL code path (#606/#1689)."""
    from concurrent.futures import ThreadPoolExecutor  # noqa: F401

    import matplotlib  # noqa: F401
    import matplotlib.pyplot as plt  # noqa: F401
    import torch  # noqa: F401
    from huggingface_hub import HfApi  # noqa: F401

    from explore_persona_space.analysis.paper_plots import (  # noqa: F401
        paper_palette_role,
        savefig_paper,
        set_paper_style,
    )
    from explore_persona_space.experiments.issue_779.fit_h import (  # noqa: F401
        ridge_fit_predict,
        ridge_fit_predict_fast_layer_batched,
    )
    from explore_persona_space.orchestrate import hub  # noqa: F401

    print("[import-check] OK: all deferred imports resolved", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--phase",
        required=True,
        choices=["stage", "verify-ridge", "run", "finalize", "figure", "import-check"],
    )
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--folds", default="", help="comma-separated fold subset (default: all)")
    ap.add_argument("--pairs", default="", help="comma-separated 'I->J' subset (default: all 12)")
    args = ap.parse_args()

    if args.phase == "import-check":
        _import_check()
        sys.stdout.flush()
        sys.exit(0)

    out_root = args.out_root
    stage_info = None
    if args.phase == "stage":
        stage_info = stage_inputs(out_root)
        R._write_json_atomic(out_root / "state" / "ladder_stage.json", stage_info)
        sys.stdout.flush()
        sys.exit(0)

    stage_path = out_root / "state" / "ladder_stage.json"
    if stage_path.exists():
        stage_info = R._read_json(stage_path)
    ckpts = list(C.CKPTS)
    ctx = LadderContext(out_root, ckpts)
    all_pairs = ordered_pairs(ckpts)
    pairs = all_pairs
    if args.pairs:
        want = {p.strip() for p in args.pairs.split(",") if p.strip()}
        pairs = [(i, j) for i, j in all_pairs if f"{i}->{j}" in want]
        if len(pairs) != len(want):
            raise SystemExit(
                f"unknown pair(s) in --pairs: {want - {f'{i}->{j}' for i, j in pairs}}"
            )
    n_folds = ctx.corpora[CORPUS].n_folds
    folds = list(range(n_folds))
    if args.folds:
        folds = [int(x) for x in args.folds.split(",") if x.strip() != ""]
        bad = [f for f in folds if f not in range(n_folds)]
        if bad:
            raise SystemExit(f"--folds out of range (n_folds={n_folds}): {bad}")

    if args.phase == "verify-ridge":
        report = verify_ridge(ctx, args.device, fold=folds[0])
        R._write_json_atomic(ctx.eval_dir / FOLLOWUP_LABEL / "ridge_parity.json", report)
        print(f"[ladder] ridge parity: {report['pass']} max_rel={report['max_rel_diff']:.3e}")
        sys.stdout.flush()
        sys.exit(0 if report["pass"] else 7)

    if args.phase == "run":
        gate_path = ctx.eval_dir / FOLLOWUP_LABEL / "ridge_parity.json"
        if not gate_path.exists() or not R._read_json(gate_path).get("pass"):
            raise SystemExit(
                f"ridge parity gate not PASSed at {gate_path} — run --phase verify-ridge first"
            )
        t0 = time.time()
        total = 0
        for fold in folds:
            total += run_fold(ctx, args.device, fold, pairs)
        print(
            f"[ladder] done: {total} units over folds {folds} in {(time.time() - t0) / 60:.1f} min",
            flush=True,
        )
        sys.stdout.flush()
        sys.exit(0)

    if args.phase == "finalize":
        result = finalize(ctx, all_pairs, stage_info)
        if not result["parity_gate"]["pass"]:
            print("[ladder] ERROR: parity gate FAILED — see parity_gate.rows", flush=True)
            sys.stdout.flush()
            sys.exit(8)  # mirrors verify-ridge's rc=7 fail-loud convention
        sys.stdout.flush()
        sys.exit(0)

    if args.phase == "figure":
        result = R._read_json(REPO_EVAL_DIR / "ladder_modes.json")
        png = make_figure(result)
        print(f"[ladder] wrote {png}", flush=True)
        sys.stdout.flush()
        sys.exit(0)


if __name__ == "__main__":
    main()
